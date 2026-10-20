from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
import queue
import random
import shutil
import threading
import time
from typing import Iterator

import numpy as np
import torch

import tomllib

_DTYPES: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "int32": np.dtype(np.int32),
    "uint8": np.dtype(np.uint8),
}
_VECTOR_FIELDS = {
    "xt",
    "x1",
    "target_internal",
    "target_center",
    "target_rotation",
}
_BOOL_FIELDS = {"valid_atom", "valid_fragment"}
_LONG_FIELDS = {
    "atom_class",
    "target_atom_class",
    "atom_index",
    "fragment_id",
    "document_id",
    "fragment_length",
    "fragment_document",
    "fragment_atom_count",
}
SUPPORTED_DOMAINS = frozenset({"qm9", "geom-drugs"})


class BatchDomainError(ValueError):
    """A valid molecular batch belongs to a different dataset."""


@dataclass
class MolecularBatch:
    xt: torch.Tensor
    x1: torch.Tensor
    target_internal: torch.Tensor
    atom_class: torch.Tensor
    target_atom_class: torch.Tensor
    atom_index: torch.Tensor
    fragment_id: torch.Tensor
    document_id: torch.Tensor
    time_token: torch.Tensor
    valid_atom: torch.Tensor
    target_center: torch.Tensor
    target_rotation: torch.Tensor
    fragment_length: torch.Tensor
    fragment_document: torch.Tensor
    fragment_atom_count: torch.Tensor
    internal_variance: torch.Tensor
    time_fragment: torch.Tensor
    valid_fragment: torch.Tensor
    metadata: dict

    @property
    def token_capacity(self) -> int:
        return self.xt.shape[0]

    @property
    def fragment_capacity(self) -> int:
        return self.fragment_length.shape[0]

    # Generic aliases keep the geometry operators concise.
    @property
    def segment_id(self) -> torch.Tensor:
        return self.fragment_id

    @property
    def segment_capacity(self) -> int:
        return self.fragment_capacity

    @property
    def segment_length(self) -> torch.Tensor:
        return self.fragment_length

    @property
    def segment_document(self) -> torch.Tensor:
        return self.fragment_document

    @property
    def segment_atom_count(self) -> torch.Tensor:
        return self.fragment_atom_count

    @property
    def time_segment(self) -> torch.Tensor:
        return self.time_fragment

    @property
    def valid_segment(self) -> torch.Tensor:
        return self.valid_fragment

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = True,
    ) -> "MolecularBatch":
        values = {
            field.name: (
                value.to(device, non_blocking=non_blocking)
                if isinstance((value := getattr(self, field.name)), torch.Tensor)
                else value
            )
            for field in fields(self)
        }
        return MolecularBatch(**values)

    def pin_memory(self) -> "MolecularBatch":
        values = {
            field.name: (
                value.pin_memory()
                if isinstance((value := getattr(self, field.name)), torch.Tensor)
                else value
            )
            for field in fields(self)
        }
        return MolecularBatch(**values)

    def record_stream(self, stream: torch.Stream) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(stream)


def _read_array(batch_dir: Path, spec: dict) -> np.ndarray:
    dtype = _DTYPES[spec["dtype"]]
    shape = tuple(int(value) for value in spec["shape"])
    array = np.fromfile(batch_dir / spec["file"], dtype=dtype)
    expected = int(np.prod(shape, dtype=np.int64))
    if array.size != expected:
        raise ValueError(
            f"{batch_dir / spec['file']} contains {array.size} values; "
            f"expected {expected}"
        )
    return array.reshape(shape, order=spec.get("order", "F"))


def _as_tensor(name: str, array: np.ndarray) -> torch.Tensor:
    if name in _VECTOR_FIELDS:
        if array.ndim != 2 or array.shape[0] != 3:
            raise ValueError(
                f"{name} must have Julia shape 3 × length, got {array.shape}"
            )
        array = np.ascontiguousarray(array.T)
    else:
        array = np.ascontiguousarray(array.reshape(-1))
    tensor = torch.from_numpy(array)
    if name in _BOOL_FIELDS:
        return tensor.bool()
    if name in _LONG_FIELDS:
        return tensor.long()
    return tensor.float()


def load_batch(
    path: str | Path,
    *,
    pin_memory: bool = False,
    expected_domain: str | None = "qm9",
) -> MolecularBatch:
    batch_dir = Path(path)
    with (batch_dir / "metadata.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    if metadata.get("format_version") != 1:
        raise ValueError(f"unsupported batch format in {batch_dir}")
    domain = metadata.get("domain")
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"unsupported molecular batch domain {domain!r}")
    if expected_domain is not None and domain != expected_domain:
        raise BatchDomainError(
            f"{batch_dir} is a {domain!r} batch, expected {expected_domain!r}"
        )
    field_specs = metadata.get("fields", {})
    expected = {field.name for field in fields(MolecularBatch)} - {"metadata"}
    if set(field_specs) != expected:
        raise ValueError(
            f"batch schema mismatch: missing={sorted(expected - set(field_specs))}, "
            f"unexpected={sorted(set(field_specs) - expected)}"
        )
    tensors = {
        name: _as_tensor(name, _read_array(batch_dir, field_specs[name]))
        for name in expected
    }
    batch = MolecularBatch(**tensors, metadata=metadata)
    return batch.pin_memory() if pin_memory else batch


def _complete_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "metadata.toml").is_file()
    )


def _claim_for_consumption(path: Path) -> bool:
    """Atomically claim a live-queue batch across consumer processes."""
    try:
        (path / ".lpgm-consumer-claim").mkdir()
    except FileExistsError:
        return False
    except FileNotFoundError:
        return False
    return True


def iter_batches(
    root: str | Path,
    *,
    live: bool = False,
    consume: bool = False,
    shuffle: bool = True,
    seed: int = 17,
    poll_seconds: float = 0.05,
    pin_memory: bool = False,
    expected_domain: str | None = "qm9",
) -> Iterator[MolecularBatch]:
    root_path = Path(root)
    rng = random.Random(seed)
    consumed: set[Path] = set()
    while True:
        paths = [path for path in _complete_dirs(root_path) if path not in consumed]
        if shuffle:
            rng.shuffle(paths)
        if not paths:
            if live:
                time.sleep(poll_seconds)
                continue
            raise ValueError(f"no unconsumed batches in {root_path}")
        for path in paths:
            if consume and not _claim_for_consumption(path):
                continue
            try:
                batch = load_batch(
                    path,
                    pin_memory=pin_memory,
                    expected_domain=expected_domain,
                )
            except BatchDomainError:
                if consume:
                    try:
                        (path / ".lpgm-consumer-claim").rmdir()
                    except FileNotFoundError:
                        pass
                raise
            except (FileNotFoundError, ValueError):
                if consume:
                    shutil.rmtree(path, ignore_errors=True)
                if live:
                    time.sleep(poll_seconds)
                    continue
                raise
            if consume:
                shutil.rmtree(path, ignore_errors=True)
            else:
                consumed.add(path)
            yield batch
        if not live:
            consumed.clear()


@dataclass
class _WorkerFailure:
    exception: BaseException


_END = object()


def _prefetch_worker(
    source: Iterator[MolecularBatch],
    output: queue.Queue,
    stop: threading.Event,
) -> None:
    try:
        for batch in source:
            while not stop.is_set():
                try:
                    output.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if stop.is_set():
                return
    except BaseException as exception:
        output.put(_WorkerFailure(exception))
        return
    output.put(_END)


class _ThreadedPrefetcher(Iterator[MolecularBatch]):
    def __init__(self, source: Iterator[MolecularBatch], *, depth: int):
        self._output: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=_prefetch_worker,
            args=(source, self._output, self._stop),
            name="mollpgm-batch-prefetch",
            daemon=True,
        )
        self._thread.start()

    def __iter__(self) -> "_ThreadedPrefetcher":
        return self

    def __next__(self) -> MolecularBatch:
        item = self._output.get()
        if item is _END:
            self.close()
            raise StopIteration
        if isinstance(item, _WorkerFailure):
            self.close()
            raise item.exception
        if not isinstance(item, MolecularBatch):
            raise TypeError(f"prefetch worker returned {type(item)!r}")
        return item

    def close(self) -> None:
        self._stop.set()


class _DevicePrefetcher(Iterator[MolecularBatch]):
    def __init__(
        self,
        source: Iterator[MolecularBatch],
        *,
        device: torch.device,
    ):
        self._source = source
        self._device = device
        self._stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._next: MolecularBatch | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            host = next(self._source)
        except StopIteration:
            self._next = None
            return
        if self._stream is None:
            self._next = host.to(self._device, non_blocking=False)
        else:
            with torch.cuda.stream(self._stream):
                self._next = host.to(self._device, non_blocking=True)

    def __iter__(self) -> "_DevicePrefetcher":
        return self

    def __next__(self) -> MolecularBatch:
        if self._next is None:
            raise StopIteration
        if self._stream is not None:
            current = torch.cuda.current_stream(self._device)
            current.wait_stream(self._stream)
            self._next.record_stream(current)
        batch = self._next
        self._preload()
        return batch


def prefetch_batches(
    source: Iterator[MolecularBatch],
    *,
    device: torch.device | str,
    depth: int = 4,
) -> Iterator[MolecularBatch]:
    if depth < 0:
        raise ValueError("prefetch depth cannot be negative")
    resolved = torch.device(device)
    if depth == 0:
        return _DevicePrefetcher(source, device=resolved)
    return _DevicePrefetcher(_ThreadedPrefetcher(source, depth=depth), device=resolved)


def clone_with_state(
    batch: MolecularBatch,
    coords: torch.Tensor,
    atom_class: torch.Tensor,
    physical_time: float,
) -> MolecularBatch:
    return replace(
        batch,
        xt=coords,
        atom_class=atom_class,
        time_token=torch.where(
            batch.valid_atom,
            torch.full_like(batch.time_token, physical_time),
            torch.zeros_like(batch.time_token),
        ),
        time_fragment=torch.where(
            batch.valid_fragment,
            torch.full_like(batch.time_fragment, physical_time),
            torch.zeros_like(batch.time_fragment),
        ),
    )
