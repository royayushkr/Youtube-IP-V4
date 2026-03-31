from __future__ import annotations

import io
import importlib
import sys
from types import ModuleType
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
def bertopic_transformers_compatibility_patch() -> Iterator[None]:
    original_values = []
    synthetic_modules = []

    try:
        modeling_bert = importlib.import_module("transformers.models.bert.modeling_bert")
    except Exception:
        modeling_bert = None

    if modeling_bert is not None:
        aliases = (
            ("BertSdpaSelfAttention", "BertSelfAttention"),
            ("BertSdpaAttention", "BertAttention"),
        )
        for alias_name, source_name in aliases:
            if hasattr(modeling_bert, alias_name) or not hasattr(modeling_bert, source_name):
                continue
            setattr(modeling_bert, alias_name, getattr(modeling_bert, source_name))
            original_values.append(("attr", modeling_bert, alias_name))

    try:
        tokenization_bert = importlib.import_module("transformers.models.bert.tokenization_bert")
    except Exception:
        tokenization_bert = None

    tokenization_fast_name = "transformers.models.bert.tokenization_bert_fast"
    if tokenization_bert is not None and tokenization_fast_name not in sys.modules:
        synthetic_module = ModuleType(tokenization_fast_name)
        tokenizer_fast = getattr(tokenization_bert, "BertTokenizerFast", None) or getattr(tokenization_bert, "BertTokenizer", None)
        tokenizer = getattr(tokenization_bert, "BertTokenizer", None)
        if tokenizer_fast is not None:
            synthetic_module.BertTokenizerFast = tokenizer_fast
        if tokenizer is not None:
            synthetic_module.BertTokenizer = tokenizer
        sys.modules[tokenization_fast_name] = synthetic_module
        synthetic_modules.append(tokenization_fast_name)

    try:
        yield
    finally:
        for kind, module_obj, alias_name in reversed(original_values):
            if kind == "attr" and hasattr(module_obj, alias_name):
                delattr(module_obj, alias_name)
        for module_name in synthetic_modules:
            sys.modules.pop(module_name, None)


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

    with bertopic_transformers_compatibility_patch():
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
