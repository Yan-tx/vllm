from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class OocBlockStore(ABC):
    @abstractmethod
    def put(self, block_id: int, block: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, block_id: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def exists(self, block_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete(self, block_id: int) -> None:
        raise NotImplementedError


class FileBlockStore(OocBlockStore):
    def __init__(self, root_dir: str, layer_key: str) -> None:
        self.layer_dir = Path(root_dir) / layer_key
        self.layer_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, block_id: int) -> Path:
        return self.layer_dir / f"block_{block_id}.pt"

    def put(self, block_id: int, block: torch.Tensor) -> None:
        torch.save(block, self._path(block_id))

    def get(self, block_id: int) -> torch.Tensor:
        return torch.load(self._path(block_id), map_location="cpu")

    def exists(self, block_id: int) -> bool:
        return self._path(block_id).exists()

    def delete(self, block_id: int) -> None:
        path = self._path(block_id)
        if path.exists():
            path.unlink()


class OocPageManager:
    def __init__(self, store: OocBlockStore, kv_cache: torch.Tensor) -> None:
        if kv_cache.device.type != "cpu":
            raise ValueError("OocPageManager requires a CPU kv_cache")
        self.store = store
        self.bind_kv_cache(kv_cache)
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._pinned: set[int] = set()

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if kv_cache.device.type != "cpu":
            raise ValueError("OocPageManager requires a CPU kv_cache")
        self.kv_cache = kv_cache
        self.capacity = kv_cache.shape[1]
        self.virtual_to_physical: dict[int, int] = {}
        self.physical_to_virtual: list[int] = [-1] * self.capacity

    def ensure_blocks(self, block_ids: Iterable[int], for_read: bool) -> None:
        unique_ids = list(dict.fromkeys(block_ids))
        if for_read and len(unique_ids) > self.capacity:
            raise RuntimeError(
                "OOC spill capacity is smaller than required blocks. "
                "Increase VLLM_OOC_KV_CPU_BLOCKS or reduce prompt length.")
        for block_id in unique_ids:
            if block_id in self.virtual_to_physical:
                self._touch(block_id)
                continue
            self._allocate_block(block_id, for_read)

    def map_block_ids(self, block_ids: Iterable[int]) -> list[int]:
        return [self.virtual_to_physical[block_id] for block_id in block_ids]

    def pin_blocks(self, block_ids: Iterable[int]) -> None:
        self._pinned.update(block_ids)

    def unpin_blocks(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            self._pinned.discard(block_id)

    def _touch(self, block_id: int) -> None:
        if block_id in self._lru:
            self._lru.pop(block_id, None)
        self._lru[block_id] = None

    def _allocate_block(self, block_id: int, for_read: bool) -> None:
        physical_id = self._get_free_physical()
        if physical_id is None:
            physical_id = self._evict_block()
        self.virtual_to_physical[block_id] = physical_id
        self.physical_to_virtual[physical_id] = block_id
        self._touch(block_id)
        if for_read:
            self._load_block(block_id, physical_id)
        else:
            self.kv_cache[:, physical_id].zero_()

    def _get_free_physical(self) -> int | None:
        for idx, virt in enumerate(self.physical_to_virtual):
            if virt == -1:
                return idx
        return None

    def _evict_block(self) -> int:
        for virt in list(self._lru.keys()):
            if virt in self._pinned:
                continue
            physical_id = self.virtual_to_physical.pop(virt)
            self._spill_block(virt, physical_id)
            self.physical_to_virtual[physical_id] = -1
            self._lru.pop(virt, None)
            return physical_id
        raise RuntimeError("No evictable blocks; all blocks are pinned")

    def _spill_block(self, block_id: int, physical_id: int) -> None:
        block = self.kv_cache[:, physical_id].detach().clone()
        self.store.put(block_id, block)

    def _load_block(self, block_id: int, physical_id: int) -> None:
        if not self.store.exists(block_id):
            raise RuntimeError(f"Missing spilled block {block_id}")
        block = self.store.get(block_id)
        if block.dtype != self.kv_cache.dtype:
            block = block.to(self.kv_cache.dtype)
        self.kv_cache[:, physical_id].copy_(block)