from __future__ import annotations

import io
from contextlib import contextmanager
from typing import Iterator


MPS_STORAGE_ERROR_PATTERNS = (
    "storage device not recognized: mps",
    "don't know how to restore data location",
)


def is_mps_storage_error(exc: Exception) -> bool:
    message = str(exc or "").strip().lower()
    if not message or "mps" not in message:
        return False
    return any(pattern in message for pattern in MPS_STORAGE_ERROR_PATTERNS)


def _call_torch_load(original_torch_load, *args, **kwargs):
    try:
        return original_torch_load(*args, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return original_torch_load(*args, **kwargs)


@contextmanager
def bertopic_cpu_compatibility_patch() -> Iterator[None]:
    import torch

    original_torch_load = torch.load
    original_default_restore_location = torch.serialization.default_restore_location
    original_load_from_bytes = getattr(torch.storage, "_load_from_bytes", None)

    def _cpu_default_restore_location(storage, location):
        normalized_location = str(location or "").strip().lower()
        if normalized_location == "mps":
            return original_default_restore_location(storage, "cpu")
        return original_default_restore_location(storage, location)

    def _cpu_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return _call_torch_load(original_torch_load, *args, **kwargs)

    def _cpu_load_from_bytes(blob):
        return _call_torch_load(original_torch_load, io.BytesIO(blob), map_location="cpu", weights_only=False)

    torch.load = _cpu_torch_load
    torch.serialization.default_restore_location = _cpu_default_restore_location
    if original_load_from_bytes is not None:
        torch.storage._load_from_bytes = _cpu_load_from_bytes

    try:
        yield
    finally:
        torch.load = original_torch_load
        torch.serialization.default_restore_location = original_default_restore_location
        if original_load_from_bytes is not None:
            torch.storage._load_from_bytes = original_load_from_bytes


def load_bertopic_with_cpu_fallback(model_path: str):
    from bertopic import BERTopic

    try:
        return BERTopic.load(model_path)
    except Exception as exc:
        if not is_mps_storage_error(exc):
            raise

    with bertopic_cpu_compatibility_patch():
        try:
            return BERTopic.load(model_path)
        except Exception as exc:
            if is_mps_storage_error(exc):
                raise RuntimeError(
                    "The BERTopic artifact still contains Apple MPS device references after CPU remapping. "
                    "Repackage and republish the model as a CPU-safe artifact bundle."
                ) from exc
            raise

