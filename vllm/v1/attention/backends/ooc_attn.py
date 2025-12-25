# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-core (streaming) attention MVP backend.

MVP constraints:
- CUDA only
- full attention (no sliding window / ALiBi)
- fp16/bf16 KV cache (GPU or CPU in OOC modes)
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
)
from vllm.attention.utils.fa_utils import reshape_and_cache_flash
from vllm.logger import init_logger
from vllm.utils import GiB_bytes
from vllm.v1.attention.backends.ooc_kv_store import (
    FileBlockStore,
    OocPageManager,
)
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class OocAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "OOC_ATTN"

    @staticmethod
    def get_impl_cls() -> type["OocAttentionImpl"]:
        return OocAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["OocAttentionMetadata"]:
        return OocAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["OocAttentionMetadataBuilder"]:
        return OocAttentionMetadataBuilder

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        # MVP: accept all head sizes.
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order


@dataclass
class OocAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class OocAttentionMetadataBuilder(AttentionMetadataBuilder[OocAttentionMetadata]):
    def __init__(
        self,
        runner: "GPUModelRunner",
        kv_cache_spec: AttentionSpec,
        block_table: BlockTable,
    ):
        self.runner = runner
        self.block_size = kv_cache_spec.block_size
        self.block_table = block_table

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> OocAttentionMetadata:
        del common_prefix_len, fast_build
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = int(self.runner.seq_lens_np[:num_reqs].max()
                          ) if num_reqs > 0 else 0
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = self.block_table
        block_table_tensor = block_table.get_device_tensor()[:num_reqs]

        block_table.slot_mapping[:num_actual_tokens].copy_(
            block_table.slot_mapping_cpu[:num_actual_tokens],
            non_blocking=True)
        # Fill unused with -1. Needed for reshape_and_cache in full cuda graph
        # mode.
        block_table.slot_mapping[num_actual_tokens:].fill_(-1)
        slot_mapping = block_table.slot_mapping[:num_actual_tokens]

        return OocAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=self.runner.query_start_loc_cpu[:num_reqs + 1],
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
        )


class OocAttentionImpl(AttentionImpl[OocAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        blocksparse_params=None,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **_extra_kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self._ooc_store_dir = os.getenv("VLLM_OOC_KV_STORE_DIR", "")
        self._spill_enabled = self._ooc_store_dir not in ("", "0", "false", "False")
        self._page_manager: OocPageManager | None = None
        self._page_manager_kv_cache_id: int | None = None
        self._layer_key: str | None = None
        self._ooc_hybrid = os.getenv("VLLM_OOC_KV_HYBRID",
                                     "") not in ("", "0", "false", "False")
        self._gpu_kv_blocks_env = os.getenv("VLLM_OOC_KV_GPU_BLOCKS")
        self._gpu_kv_gb_env = os.getenv("VLLM_OOC_KV_GPU_GB")
        self._gpu_kv_cache: torch.Tensor | None = None
        self._gpu_kv_blocks = 0
        self._gpu_kv_cache_shape: tuple[int, ...] | None = None

        if alibi_slopes is not None:
            raise NotImplementedError("OOC_ATTN MVP does not support ALiBi.")
        if sliding_window is not None:
            raise NotImplementedError(
                "OOC_ATTN MVP does not support sliding window attention."
            )
        if blocksparse_params is not None:
            raise NotImplementedError(
                "OOC_ATTN MVP does not support block-sparse attention."
            )
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "OOC_ATTN MVP does not support logits soft cap."
            )

    def _resolve_gpu_kv_blocks(
        self,
        num_blocks_total: int,
        block_size: int,
        dtype: torch.dtype,
    ) -> int:
        if not self._ooc_hybrid:
            return 0
        if self._gpu_kv_blocks_env:
            try:
                blocks = int(self._gpu_kv_blocks_env)
            except ValueError as exc:
                raise ValueError(
                    "VLLM_OOC_KV_GPU_BLOCKS must be an integer") from exc
        elif self._gpu_kv_gb_env:
            try:
                gpu_gb = float(self._gpu_kv_gb_env)
            except ValueError as exc:
                raise ValueError(
                    "VLLM_OOC_KV_GPU_GB must be a number") from exc
            bytes_per_block = (
                2 * block_size * self.num_kv_heads * self.head_size *
                torch.tensor([], dtype=dtype).element_size())
            if bytes_per_block <= 0:
                return 0
            blocks = int((gpu_gb * GiB_bytes) // bytes_per_block)
        else:
            return 0

        if blocks < 0:
            blocks = 0
        if blocks > num_blocks_total:
            blocks = num_blocks_total
        return blocks

    def _ensure_gpu_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if not self._ooc_hybrid or kv_cache.device.type != "cpu":
            return
        num_blocks_total = kv_cache.shape[1]
        block_size = kv_cache.shape[2]
        num_kv_heads = kv_cache.shape[3]
        head_size = kv_cache.shape[4]
        dtype = kv_cache.dtype
        gpu_blocks = self._resolve_gpu_kv_blocks(num_blocks_total, block_size,
                                                 dtype)
        if gpu_blocks <= 0:
            self._gpu_kv_cache = None
            self._gpu_kv_blocks = 0
            self._gpu_kv_cache_shape = None
            return
        desired_shape = (2, gpu_blocks, block_size, num_kv_heads, head_size)
        needs_alloc = (self._gpu_kv_cache is None
                       or self._gpu_kv_cache_shape != desired_shape
                       or self._gpu_kv_cache.dtype != dtype)
        if needs_alloc:
            self._gpu_kv_cache = torch.empty(
                desired_shape,
                device="cuda",
                dtype=dtype,
            )
            self._gpu_kv_cache.copy_(
                kv_cache[:, :gpu_blocks].to(device="cuda", non_blocking=True))
            self._gpu_kv_blocks = gpu_blocks
            self._gpu_kv_cache_shape = desired_shape

    def _get_layer_key(self, layer) -> str:
        if self._layer_key is not None:
            return self._layer_key
        layer_key = getattr(layer, "_ooc_layer_key", None)
        if layer_key:
            self._layer_key = layer_key
            return layer_key
        layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx is None:
            layer_idx = getattr(layer, "layer_id", None)
        if layer_idx is None:
            layer_idx = id(layer)
        layer_key = f"layer_{layer_idx}_pid_{os.getpid()}"
        self._layer_key = layer_key
        try:
            setattr(layer, "_ooc_layer_key", layer_key)
        except Exception:
            pass
        return layer_key

    def _maybe_init_page_manager(self, layer, kv_cache: torch.Tensor) -> None:
        if not self._spill_enabled:
            return
        if kv_cache.device.type != "cpu":
            return
        cache_id = id(kv_cache)
        existing = getattr(layer, "_ooc_page_manager", None)
        existing_id = getattr(layer, "_ooc_page_manager_kv_cache_id", None)
        if existing is not None and existing_id == cache_id:
            self._page_manager = existing
            self._page_manager_kv_cache_id = cache_id
            return
        if (self._page_manager is None
                or self._page_manager_kv_cache_id != cache_id):
            store = FileBlockStore(self._ooc_store_dir,
                                   self._get_layer_key(layer))
            self._page_manager = OocPageManager(store, kv_cache)
            self._page_manager_kv_cache_id = cache_id
            try:
                setattr(layer, "_ooc_page_manager", self._page_manager)
                setattr(layer, "_ooc_page_manager_kv_cache_id", cache_id)
            except Exception:
                pass

    @staticmethod
    def _write_kv_cache_cpu(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
        gpu_kv_cache: torch.Tensor | None = None,
        gpu_blocks: int = 0,
        page_manager: OocPageManager | None = None,
    ) -> None:
        slot_mapping_cpu = slot_mapping
        if slot_mapping_cpu.device.type != "cpu":
            slot_mapping_cpu = slot_mapping_cpu.to("cpu")
        if slot_mapping_cpu.numel() == 0:
            return

        key_cpu = key.detach().to("cpu", non_blocking=True)
        value_cpu = value.detach().to("cpu", non_blocking=True)
        if slot_mapping_cpu.numel() < key_cpu.shape[0]:
            padded = torch.full((key_cpu.shape[0],), -1,
                                dtype=slot_mapping_cpu.dtype)
            padded[:slot_mapping_cpu.numel()] = slot_mapping_cpu
            slot_mapping_cpu = padded
        key_gpu = key
        value_gpu = value

        if (slot_mapping_cpu < 0).any():
            valid = slot_mapping_cpu >= 0
            if not valid.any():
                return
            slot_mapping_cpu = slot_mapping_cpu[valid]
            key_cpu = key_cpu[valid]
            value_cpu = value_cpu[valid]
            key_gpu = key_gpu[valid]
            value_gpu = value_gpu[valid]

        block_ids = slot_mapping_cpu // block_size
        block_offsets = slot_mapping_cpu % block_size

        virtual_ids = None
        if page_manager is not None:
            virtual_ids = block_ids.tolist()
            unique_ids = list(dict.fromkeys(virtual_ids))
            for virt_id in unique_ids:
                page_manager.ensure_blocks([virt_id], for_read=False)
                page_manager.pin_blocks([virt_id])
                try:
                    phys_id = page_manager.map_block_ids([virt_id])[0]
                    idxs = [i for i, v in enumerate(virtual_ids) if v == virt_id]
                    idx_tensor = torch.tensor(idxs, dtype=torch.long)
                    key_cache[phys_id, block_offsets[idx_tensor]] = key_cpu[idx_tensor]
                    value_cache[phys_id, block_offsets[idx_tensor]] = value_cpu[idx_tensor]
                finally:
                    page_manager.unpin_blocks([virt_id])
        else:
            key_cache[block_ids, block_offsets] = key_cpu
            value_cache[block_ids, block_offsets] = value_cpu

        if (gpu_kv_cache is not None and gpu_blocks > 0
                and page_manager is None):
            gpu_mask = block_ids < gpu_blocks
            if gpu_mask.any():
                gpu_idx = torch.nonzero(gpu_mask, as_tuple=False).squeeze(-1)
                block_ids_gpu = block_ids[gpu_idx].to(device=key.device,
                                                      non_blocking=True)
                block_offsets_gpu = block_offsets[gpu_idx].to(
                    device=key.device, non_blocking=True)
                gpu_idx_device = gpu_idx.to(device=key.device,
                                            non_blocking=True)
                key_gpu_sel = key_gpu.index_select(0, gpu_idx_device)
                value_gpu_sel = value_gpu.index_select(0, gpu_idx_device)
                gpu_kv_cache[0][block_ids_gpu, block_offsets_gpu] = key_gpu_sel
                gpu_kv_cache[1][block_ids_gpu,
                                block_offsets_gpu] = value_gpu_sel

    def _gather_kv_blocks_hybrid(
        self,
        key_cache_cpu: torch.Tensor,
        value_cache_cpu: torch.Tensor,
        block_ids_cpu: torch.Tensor,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._gpu_kv_cache is None or self._gpu_kv_blocks <= 0:
            k = key_cache_cpu.index_select(0, block_ids_cpu).reshape(
                block_ids_cpu.numel() * block_size,
                self.num_kv_heads,
                self.head_size,
            )
            v = value_cache_cpu.index_select(0, block_ids_cpu).reshape(
                block_ids_cpu.numel() * block_size,
                self.num_kv_heads,
                self.head_size,
            )
            return (k.to(device, non_blocking=True),
                    v.to(device, non_blocking=True))

        num_blocks = block_ids_cpu.numel()
        k_out = torch.empty(
            (num_blocks, block_size, self.num_kv_heads, self.head_size),
            device=device,
            dtype=dtype,
        )
        v_out = torch.empty_like(k_out)

        gpu_mask = block_ids_cpu < self._gpu_kv_blocks
        if gpu_mask.any():
            gpu_pos = torch.nonzero(gpu_mask, as_tuple=False).squeeze(-1)
            gpu_pos_device = gpu_pos.to(device=device, non_blocking=True)
            gpu_block_ids = block_ids_cpu[gpu_pos].to(device=device,
                                                      non_blocking=True)
            k_out[gpu_pos_device] = self._gpu_kv_cache[0].index_select(
                0, gpu_block_ids)
            v_out[gpu_pos_device] = self._gpu_kv_cache[1].index_select(
                0, gpu_block_ids)

        cpu_mask = ~gpu_mask
        if cpu_mask.any():
            cpu_pos = torch.nonzero(cpu_mask, as_tuple=False).squeeze(-1)
            cpu_block_ids = block_ids_cpu[cpu_pos]
            cpu_k = key_cache_cpu.index_select(0, cpu_block_ids)
            cpu_v = value_cache_cpu.index_select(0, cpu_block_ids)
            cpu_pos_device = cpu_pos.to(device=device, non_blocking=True)
            k_out[cpu_pos_device] = cpu_k.to(device, non_blocking=True)
            v_out[cpu_pos_device] = cpu_v.to(device, non_blocking=True)

        return k_out, v_out

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: OocAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None:
            raise NotImplementedError(
                "OOC_ATTN MVP does not support fused output quantization."
            )

        if attn_metadata is None:
            return output

        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "OOC_ATTN MVP only supports decoder attention."
            )

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "OOC_ATTN MVP does not support FP8 KV cache."
            )

        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"OOC_ATTN MVP only supports fp16/bf16 queries, got {query.dtype}."
            )
        if query.device.type != "cuda":
            raise NotImplementedError("OOC_ATTN MVP only supports CUDA.")

        kv_on_cpu = kv_cache.device.type == "cpu"
        use_spill = kv_on_cpu and self._spill_enabled
        if use_spill:
            self._maybe_init_page_manager(layer, kv_cache)
            if self._page_manager is None:
                logger.warning("OOC spill store enabled but page manager is not available.")
                use_spill = False

        use_hybrid = kv_on_cpu and self._ooc_hybrid and not use_spill
        if use_spill and self._ooc_hybrid:
            logger.info_once("OOC spill store enabled; disabling GPU mirror.")
        if use_hybrid:
            self._ensure_gpu_kv_cache(kv_cache)
            if self._gpu_kv_blocks <= 0:
                use_hybrid = False

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            if kv_on_cpu:
                key_cache_cpu, value_cache_cpu = kv_cache.unbind(0)
                self._write_kv_cache_cpu(
                    key_cache_cpu,
                    value_cache_cpu,
                    key,
                    value,
                    attn_metadata.slot_mapping,
                    key_cache_cpu.shape[1],
                    gpu_kv_cache=self._gpu_kv_cache if use_hybrid else None,
                    gpu_blocks=self._gpu_kv_blocks if use_hybrid else 0,
                    page_manager=self._page_manager if use_spill else None,
                )
            else:
                reshape_and_cache_flash(
                    key,
                    value,
                    kv_cache[0],
                    kv_cache[1],
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        key_cache, value_cache = kv_cache.unbind(0)
        if key_cache.dtype != query.dtype:
            raise NotImplementedError(
                "OOC_ATTN MVP requires KV cache dtype to match query dtype."
            )

        num_reqs = attn_metadata.seq_lens.shape[0]
        if num_reqs == 0:
            return output

        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        block_size = key_cache.shape[1]
        query_start_loc_cpu = attn_metadata.query_start_loc_cpu

        output.fill_(0)

        for req_idx in range(num_reqs):
            seq_len = int(seq_lens[req_idx].item())
            if seq_len == 0:
                continue
            query_start = int(query_start_loc_cpu[req_idx].item())
            query_end = int(query_start_loc_cpu[req_idx + 1].item())
            query_len = query_end - query_start
            if query_len <= 0:
                continue
            context_len = seq_len - query_len
            if context_len < 0:
                raise ValueError(
                    "OOC_ATTN metadata error: context_len is negative."
                )

            num_blocks = (seq_len + block_size - 1) // block_size
            if kv_on_cpu:
                block_ids = block_table[req_idx, :num_blocks].to(
                    "cpu", dtype=torch.long)
            else:
                block_ids = block_table[req_idx, :num_blocks].to(torch.long)

            if kv_on_cpu and use_spill:
                virtual_ids = block_ids.tolist()
                page_manager = self._page_manager
                page_manager.ensure_blocks(virtual_ids, for_read=True)
                page_manager.pin_blocks(virtual_ids)
                try:
                    physical_ids = page_manager.map_block_ids(virtual_ids)
                    block_ids = torch.tensor(physical_ids, dtype=torch.long)
                    k = key_cache.index_select(0, block_ids).reshape(
                        num_blocks * block_size, self.num_kv_heads,
                        self.head_size)[:seq_len]
                    v = value_cache.index_select(0, block_ids).reshape(
                        num_blocks * block_size, self.num_kv_heads,
                        self.head_size)[:seq_len]
                    k = k.to(query.device, non_blocking=True)
                    v = v.to(query.device, non_blocking=True)
                finally:
                    page_manager.unpin_blocks(virtual_ids)
            elif kv_on_cpu and use_hybrid:
                k_blocks, v_blocks = self._gather_kv_blocks_hybrid(
                    key_cache,
                    value_cache,
                    block_ids,
                    block_size,
                    query.device,
                    query.dtype,
                )
                k = k_blocks.reshape(
                    num_blocks * block_size, self.num_kv_heads, self.head_size
                )[:seq_len]
                v = v_blocks.reshape(
                    num_blocks * block_size, self.num_kv_heads, self.head_size
                )[:seq_len]
            else:
                k = key_cache.index_select(0, block_ids).reshape(
                    num_blocks * block_size,
                    self.num_kv_heads,
                    self.head_size,
                )[:seq_len]
                v = value_cache.index_select(0, block_ids).reshape(
                    num_blocks * block_size,
                    self.num_kv_heads,
                    self.head_size,
                )[:seq_len]

                if kv_on_cpu:
                    k = k.to(query.device, non_blocking=True)
                    v = v.to(query.device, non_blocking=True)

            k_f = k.to(torch.float32)
            v_f = v.to(torch.float32)

            for q_offset in range(query_len):
                query_idx = query_start + q_offset
                q = query[query_idx]
                max_k = context_len + q_offset + 1

                q_f = q.to(torch.float32)
                k_allowed = k_f[:max_k]
                v_allowed = v_f[:max_k]

                q_f = q_f.view(self.num_kv_heads, self.num_queries_per_kv,
                               self.head_size)
                logits = torch.einsum("gqh,tgh->gqt", q_f, k_allowed) * self.scale
                attn = torch.softmax(logits, dim=-1)
                out = torch.einsum("gqt,tgh->gqh", attn, v_allowed)
                out = out.reshape(self.num_heads, self.head_size).to(query.dtype)
                output[query_idx] = out

        return output