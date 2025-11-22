"""High-level ingestion helpers for the binary module.

This module exposes a simple entry-point that accepts arbitrary payloads,
normalises them to bytes, binarises the data via ``binary.binarify_data``,
extracts chunk metadata, registers the dataset, and exposes convenience
wrappers for downstream access.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Union
import io
import json

import numpy as np

import binary


@dataclass
class LoadResult:
    """Container returned by ``load_payload``."""

    registry: MutableMapping[str, Any]
    entry: MutableMapping[str, Any]
    manifest: Mapping[str, Any]
    chunk_metadata: Iterable[Mapping[str, Any]]
    features: np.ndarray
    source_metadata: Mapping[str, Any]
    element_records: Iterable[Mapping[str, Any]]  # Element-level byte records


def _coerce_to_bytes(payload: Any, *, encoding: str = "utf-8") -> Tuple[bytes, Dict[str, Any]]:
    """Coerce ``payload`` to bytes while collecting provenance metadata.

    Parameters
    ----------
    payload:
        Arbitrary Python object containing the dataset to ingest.
    encoding:
        Fallback text encoding when converting textual payloads.

    Returns
    -------
    Tuple[bytes, Dict[str, Any]]
        Binary representation and descriptive metadata.

    Raises
    ------
    TypeError
        If the payload cannot be coerced into bytes.
    """

    meta: Dict[str, Any] = {
        "encoding": encoding,
        "original_type": type(payload).__name__,
    }

    if payload is None:
        raise TypeError("payload must not be None")

    if isinstance(payload, bytes):
        return payload, meta

    if isinstance(payload, bytearray):
        return bytes(payload), meta

    if isinstance(payload, memoryview):
        return payload.tobytes(), meta

    if isinstance(payload, str):
        meta["encoding_used"] = encoding
        return payload.encode(encoding), meta

    if isinstance(payload, Path):
        data = payload.read_bytes()
        meta["path"] = str(payload)
        return data, meta

    if isinstance(payload, io.IOBase):
        data = payload.read()
        if isinstance(data, str):
            meta["encoding_used"] = encoding
            data = data.encode(encoding)
        elif not isinstance(data, bytes):
            data = bytes(data)
        meta["stream_position"] = getattr(payload, "tell", lambda: None)()
        return data, meta

    if isinstance(payload, np.ndarray):
        meta.update({
            "array_shape": payload.shape,
            "array_dtype": str(payload.dtype),
        })
        return payload.tobytes(), meta

    if hasattr(payload, "tobytes") and callable(payload.tobytes):
        data = payload.tobytes()
        return data, meta

    if hasattr(payload, "read") and callable(payload.read):
        data = payload.read()
        if isinstance(data, str):
            meta["encoding_used"] = encoding
            data = data.encode(encoding)
        elif not isinstance(data, bytes):
            data = bytes(data)
        return data, meta

    if isinstance(payload, (Mapping, list, tuple, set)):
        meta["serialization"] = "json"
        return json.dumps(payload, ensure_ascii=False).encode(encoding), meta

    try:
        buffer = memoryview(payload)
    except TypeError:
        buffer = None

    if buffer is not None:
        return buffer.tobytes(), meta

    try:
        meta["serialization"] = "repr"
        return repr(payload).encode(encoding), meta
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported payload type: {type(payload)!r}") from exc


def load_payload(
    payload: Any,
    dataset_id: str,
    *,
    registry: Optional[MutableMapping[str, Any]] = None,
    chunk_size: int = 2048,
    overlap: int = 0,
    base_chunk: int = binary.BASE_CHUNK,
    encoding: str = "utf-8",
) -> LoadResult:
    """Ingest ``payload`` and register it in the reconstruction registry.

    Steps:
    1. Coerce ``payload`` to bytes while capturing provenance metadata.
    2. Binarise the bytes with :func:`binary.binarify_data`.
    3. Derive chunk-level features via :func:`binary.extract_4d_features`.
    4. Register (or reuse) a registry using :func:`binary.register_dataset`.

    Returns a :class:`LoadResult` with registry and helper artefacts.
    """

    if not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")

    data_bytes, source_meta = _coerce_to_bytes(payload, encoding=encoding)

    manifest = binary.binarify_data(
        data_bytes,
        chunk_size=chunk_size,
        overlap=overlap,
        base_chunk=base_chunk,
    )

    features, chunk_metadata = binary.extract_4d_features(manifest, data_bytes)

    # Extract all element-level records from chunks
    element_records = []
    for chunk in manifest['chunks']:
        chunk_elements = chunk.get('elements', [])
        # Enhance with dataset_id for global uniqueness
        for elem in chunk_elements:
            elem_record = dict(elem)
            elem_record['dataset_id'] = dataset_id
            elem_record['element_signature'] = f"{dataset_id}:{elem.get('element_signature')}"
            # Ensure absolute_offset exists and is an int. If missing, try to compute
            # it from chunk start + element_index using the manifest information.
            try:
                ao = elem_record.get('absolute_offset')
                if ao is None:
                    # attempt to compute from chunk index and element_index
                    chunk_idx = elem_record.get('chunk_index') if elem_record.get('chunk_index') is not None else elem_record.get('chunk_idx')
                    elem_idx = elem_record.get('element_index') if elem_record.get('element_index') is not None else elem_record.get('element_idx')
                    if chunk_idx is not None and elem_idx is not None:
                        # find chunk start in manifest
                        chunk_start = None
                        for cm in manifest.get('chunks', []):
                            # manifest chunk may use 'index' or 'chunk_index'
                            if cm.get('index') == chunk_idx or cm.get('chunk_index') == chunk_idx:
                                chunk_start = cm.get('start') if cm.get('start') is not None else cm.get('start_position')
                                break
                        # fallback: if chunk_idx is an integer and within range
                        if chunk_start is None:
                            try:
                                ci = int(chunk_idx)
                                chunks_list = manifest.get('chunks', [])
                                if 0 <= ci < len(chunks_list):
                                    cm = chunks_list[ci]
                                    chunk_start = cm.get('start') if cm.get('start') is not None else cm.get('start_position')
                            except Exception:
                                chunk_start = None
                        if chunk_start is not None:
                            elem_record['absolute_offset'] = int(chunk_start) + int(elem_idx)
                        else:
                            elem_record['absolute_offset'] = None
                else:
                    # coerce to int when possible
                    try:
                        elem_record['absolute_offset'] = int(ao)
                    except Exception:
                        elem_record['absolute_offset'] = None
            except Exception:
                elem_record['absolute_offset'] = None

            element_records.append(elem_record)

    active_registry = registry or binary.create_reconstruction_registry()
    entry = binary.register_dataset(active_registry, dataset_id, manifest, chunk_metadata)

    entry.setdefault("ingest_metadata", {})
    entry["ingest_metadata"].update({
        "source": source_meta,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "base_chunk": base_chunk,
        "feature_dimensions": features.shape,
        "total_elements": len(element_records),
    })

    # Store element records in registry entry for easy access
    entry['element_records'] = element_records

    # Validate element_records offsets and log summary warnings if issues found
    try:
        bad = [e for e in element_records if not isinstance(e.get('absolute_offset'), int)]
        if bad:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "load_payload: %d element records missing valid absolute_offset for dataset %s; first sample=%s",
                len(bad), dataset_id, (bad[0] if len(bad) <= 5 else bad[:5])
            )
    except Exception:
        pass

    return LoadResult(
        registry=active_registry,
        entry=entry,
        manifest=manifest,
        chunk_metadata=chunk_metadata,
        features=features,
        source_metadata=source_meta,
        element_records=element_records,
    )


def get_chunk_metadata(
    registry: Mapping[str, Any],
    dataset_id: str,
    chunk_index: int,
) -> Mapping[str, Any]:
    """Proxy to :func:`binary.get_chunk_metadata`."""

    return binary.get_chunk_metadata(registry, dataset_id, chunk_index)


def get_byte_metadata(
    registry: Mapping[str, Any],
    dataset_id: str,
    byte_offset: int,
) -> Mapping[str, Any]:
    """Proxy to :func:`binary.get_byte_metadata`."""

    return binary.get_byte_metadata(registry, dataset_id, byte_offset)


def reconstruct_chunk(
    registry: Mapping[str, Any],
    dataset_id: str,
    chunk_index: int,
) -> bytes:
    """Proxy to :func:`binary.reconstruct_chunk_from_registry`."""

    return binary.reconstruct_chunk_from_registry(registry, dataset_id, chunk_index)


def reconstruct_range(
    registry: Mapping[str, Any],
    dataset_id: str,
    start_byte: int,
    end_byte: Optional[int] = None,
) -> bytes:
    """Proxy to :func:`binary.reconstruct_byte_range`."""

    return binary.reconstruct_byte_range(registry, dataset_id, start_byte, end_byte)


def get_dataset_entry(
    registry: Mapping[str, Any],
    dataset_id: str,
) -> Mapping[str, Any]:
    """Proxy to :func:`binary.get_dataset_entry`."""

    return binary.get_dataset_entry(registry, dataset_id)


def reconstruct_dataset(
    registry: Mapping[str, Any],
    dataset_id: str,
) -> bytes:
    """Proxy to :func:`binary.reconstruct_dataset_from_registry`."""

    return binary.reconstruct_dataset_from_registry(registry, dataset_id)


def get_element_records(
    registry: Mapping[str, Any],
    dataset_id: str,
) -> Iterable[Mapping[str, Any]]:
    """Get all element-level records for a dataset."""
    entry = binary.get_dataset_entry(registry, dataset_id)
    return entry.get('element_records', [])


def get_element_by_offset(
    registry: Mapping[str, Any],
    dataset_id: str,
    absolute_offset: int,
) -> Mapping[str, Any]:
    """Get element record for a specific byte offset."""
    element_records = get_element_records(registry, dataset_id)
    for elem in element_records:
        if elem['absolute_offset'] == absolute_offset:
            return elem
    raise IndexError(f"No element found at offset {absolute_offset} in dataset '{dataset_id}'")


def get_elements_in_range(
    registry: Mapping[str, Any],
    dataset_id: str,
    start_offset: int,
    end_offset: int,
) -> Iterable[Mapping[str, Any]]:
    """Get all element records within a byte range [start_offset, end_offset)."""
    element_records = get_element_records(registry, dataset_id)
    # Be defensive: some element records may have missing or non-int absolute_offset
    results = []
    for elem in element_records:
        ao = elem.get('absolute_offset')
        try:
            if isinstance(ao, int) and start_offset <= ao < end_offset:
                results.append(elem)
        except Exception:
            # Skip malformed records rather than raising
            continue
    return results


__all__ = [
    "LoadResult",
    "load_payload",
    "get_chunk_metadata",
    "get_byte_metadata",
    "reconstruct_chunk",
    "reconstruct_range",
    "get_dataset_entry",
    "reconstruct_dataset",
    "get_element_records",
    "get_element_by_offset",
    "get_elements_in_range",
]
