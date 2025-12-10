import os
import math
import tempfile
import logging
from typing import Any, Dict
from collections import OrderedDict

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from matrixtransformer import MatrixTransformer as mt, MatrixType


class hyper_conn(mt):
    """
    Hyperdimensional connection finder that extends MatrixTransformer.
    
    This class provides methods for finding connections in hyperdimensional space
    between matrices and tensors, with support for memory-backed processing and
    candidate pre-filtering.
    """
    
    def __init__(self, dimensions=None, matrix_types=None):
        """
        Initialize the hyper_conn class.
        
        Args:
            dimensions: Optional dimension parameter for the transformer (default: 256)
            matrix_types: Optional list of matrix types to use
        """
        # Call parent class constructor
        super().__init__(dimensions=dimensions, matrix_types=matrix_types)
        
        # Initialize hyper_conn specific attributes
        self.hyperdimensional_connections = {}
        self._content_preview_cache = OrderedDict()
        self._preview_cache_maxsize = 128
        self._projection_norms = []
        self._projection_norm_mean = 0.0
        self._projection_norm_std = 0.0

    def _prepare_connection_context(
        self,
        num_dims,
        min_similarity,
        min_ratio,
        top_k,
        registry,
        dataset_id,
        matrix_to_chunk_map,
        include_element_metadata,
        preview_size,
        preview_cache_size,
    ):
        """Normalize inputs and build shared state for the connection search."""
        matrix_to_chunk_map = matrix_to_chunk_map or {}
        if not isinstance(matrix_to_chunk_map, dict):
            try:
                matrix_to_chunk_map = {idx: chunk_idx for idx, chunk_idx in enumerate(matrix_to_chunk_map)}
            except Exception:
                matrix_to_chunk_map = {}

        # Ensure preview cache is initialized before reuse
        if not hasattr(self, '_content_preview_cache') or self._content_preview_cache is None:
            self._content_preview_cache = OrderedDict()
        self._preview_cache_maxsize = preview_cache_size

        connection_limit = top_k if isinstance(top_k, int) and top_k > 0 else None

        context = {
            'num_dims': num_dims,
            'min_similarity': float(min_similarity),
            'min_ratio': float(min_ratio),
            'top_k': top_k,
            'connection_limit': connection_limit,
            'registry': registry,
            'dataset_id': dataset_id,
            'matrix_to_chunk_map': matrix_to_chunk_map,
            'include_element_metadata': include_element_metadata,
            'preview_size': preview_size,
            'preview_cache_size': preview_cache_size,
            'semantic_metadata_cache': {},
            'per_chunk_full_connections': None,
            'cid_index': {},
            'per_chunk_metadata_dir': None,
        }

        from collections import defaultdict

        context['per_chunk_full_connections'] = defaultdict(list)
        per_chunk_metadata_dir = f"element_metadata_{dataset_id}" if dataset_id else "element_metadata"
        if not os.path.exists(per_chunk_metadata_dir):
            try:
                os.makedirs(per_chunk_metadata_dir)
            except Exception:
                pass
        context['per_chunk_metadata_dir'] = per_chunk_metadata_dir

        return context

    def _generate_content_preview(self, elements_list, load_module, context, preview_size_local=None):
        """Reconstruct content previews with caching support."""
        if not elements_list:
            return ""

        registry = context['registry']
        dataset_id = context['dataset_id']
        preview_size_local = preview_size_local or context['preview_size']

        source_meta: Dict[str, Any] = {}
        original_type = "bytes"
        encoding = "utf-8"
        try:
            if registry is not None and dataset_id is not None:
                entry = load_module.get_dataset_entry(registry, dataset_id)
                source_meta = entry.get('ingest_metadata', {}).get('source', {}) or {}
                original_type = source_meta.get('original_type', original_type) or original_type
                encoding = source_meta.get('encoding_used') or source_meta.get('encoding') or encoding
        except Exception:
            source_meta = {}
            original_type = 'bytes'

        offsets = []
        for elem in elements_list:
            try:
                off = int(elem.get('absolute_offset', elem.get('offset', 0)))
                length = int(elem.get('length', elem.get('length_bytes', elem.get('byte_length', 1) or 1)))
                offsets.append((off, length))
            except Exception:
                continue

        if not offsets:
            byte_values = [int(v) for v in (elem.get('value') for elem in elements_list) if v is not None]
            if byte_values:
                try:
                    preview_bytes = bytes(byte_values)
                    try:
                        return self._cache_preview_result(preview_bytes.decode(encoding, errors='replace'), None, preview_size_local, context)
                    except Exception:
                        return self._cache_preview_result(preview_bytes.hex()[:preview_size_local * 2], None, preview_size_local, context)
                except Exception:
                    return ""

        offsets.sort(key=lambda x: x[0])
        merged_ranges = []
        for off, length in offsets:
            if not merged_ranges:
                merged_ranges.append([off, length])
                continue
            last_off, last_len = merged_ranges[-1]
            if off <= last_off + last_len:
                new_len = max(last_off + last_len, off + length) - last_off
                merged_ranges[-1][1] = new_len
            else:
                merged_ranges.append([off, length])

        cache_key = None
        try:
            chunk_idxs = [elem.get('chunk_index') or elem.get('chunk_idx') for elem in elements_list]
            chunk_idxs_norm = [int(ci) if ci is not None else None for ci in chunk_idxs]
            unique_chunk_idxs = set(chunk_idxs_norm)
            if len(unique_chunk_idxs) == 1 and list(unique_chunk_idxs)[0] is not None:
                cache_key = ('chunk', dataset_id, int(list(unique_chunk_idxs)[0]), preview_size_local)
            elif merged_ranges:
                start_range, len_range = merged_ranges[0]
                cache_key = ('range', dataset_id, int(start_range), int(start_range + len_range), preview_size_local)
        except Exception:
            cache_key = None

        try:
            cache = getattr(self, '_content_preview_cache', None)
            if cache_key is not None and cache is not None and cache_key in cache:
                try:
                    cache.move_to_end(cache_key)
                except Exception:
                    pass
                logging.debug(f"Preview cache hit for key {cache_key}")
                return cache[cache_key]
        except Exception:
            pass

        total_preview_bytes = 0
        snippets = []
        for start, length in merged_ranges:
            if total_preview_bytes >= preview_size_local:
                break
            end = start + length
            if total_preview_bytes + length > preview_size_local:
                end = start + (preview_size_local - total_preview_bytes)
            block = b""
            try:
                block = load_module.reconstruct_range(registry, dataset_id, start, end)
                if not isinstance(block, (bytes, bytearray)):
                    block = bytes(block)
            except Exception:
                try:
                    chunk_idx = elements_list[0].get('chunk_index') or elements_list[0].get('chunk_idx')
                    if chunk_idx is not None:
                        block = load_module.reconstruct_chunk(registry, dataset_id, int(chunk_idx))
                except Exception:
                    block = b""

            if not block:
                continue

            take = min(len(block), preview_size_local - total_preview_bytes)
            snippets.append(block[:take])
            total_preview_bytes += take

        preview_bytes = b"".join(snippets)[:preview_size_local]

        import json

        def safe_text_decode(data: bytes):
            try:
                return data.decode(encoding, errors='replace')
            except Exception:
                try:
                    return data.decode('utf-8', errors='replace')
                except Exception:
                    return None

        result = self._try_cache_preview_json(preview_bytes, source_meta, original_type, cache_key, preview_size_local, context, safe_text_decode)
        if result is not None:
            return result

        if original_type in {'str', 'string', 'Text'}:
            txt = safe_text_decode(preview_bytes)
            if txt:
                truncated = txt[:preview_size_local] + ('...' if len(txt) > preview_size_local else '')
                return self._cache_preview_result(truncated, cache_key, preview_size_local, context)

        try:
            if source_meta.get('array_shape') or source_meta.get('array_dtype') or original_type in {'ndarray', 'ndarray_like'}:
                dtype = source_meta.get('array_dtype')
                shape = source_meta.get('array_shape')
                dtype_obj = np.dtype(dtype) if dtype else None
                if dtype_obj is not None and shape:
                    shape_tuple = tuple(shape) if isinstance(shape, (list, tuple)) else (int(shape),)
                    arr = np.frombuffer(preview_bytes, dtype=dtype_obj)
                    if arr.size > 0:
                        try:
                            arr = arr[:int(np.prod(shape_tuple))]
                            arr = arr.reshape(shape_tuple)
                        except Exception:
                            pass
                        snippet = np.array2string(arr, threshold=10, max_line_width=preview_size_local)[:preview_size_local]
                        return self._cache_preview_result(snippet, cache_key, preview_size_local, context)
        except Exception:
            pass

        try:
            import imghdr
            itype = imghdr.what(None, preview_bytes)
            if itype:
                try:
                    from PIL import Image
                    import io as _io
                    import base64 as _base64
                    img = Image.open(_io.BytesIO(preview_bytes))
                    img.thumbnail((64, 64))
                    buf = _io.BytesIO()
                    img.save(buf, format='PNG')
                    b64 = _base64.b64encode(buf.getvalue()).decode('ascii')
                    payload = f"<image: {itype}, {len(preview_bytes)} bytes, thumbnail:data:image/png;base64,{b64}>"
                    return self._cache_preview_result(payload, cache_key, preview_size_local, context)
                except Exception:
                    return self._cache_preview_result(f"<image: {itype}, {len(preview_bytes)} bytes>", cache_key, preview_size_local, context)
        except Exception:
            pass

        txt = safe_text_decode(preview_bytes)
        if txt:
            sanitized = ''.join(ch if ch.isprintable() or ch.isspace() else '�' for ch in txt)
            truncated = sanitized[:preview_size_local]
            if len(sanitized) > preview_size_local:
                truncated += '...'
            return self._cache_preview_result(truncated, cache_key, preview_size_local, context)

        try:
            payload = f"<binary: {preview_bytes[:min(len(preview_bytes), 64)].hex()}... ({len(preview_bytes)} bytes)>"
            return self._cache_preview_result(payload, cache_key, preview_size_local, context)
        except Exception:
            return self._cache_preview_result(f"<binary: {len(preview_bytes)} bytes>", cache_key, preview_size_local, context)

    def _cache_preview_result(self, value, cache_key, preview_size_local, context):
        try:
            cache = getattr(self, '_content_preview_cache', None)
            if cache is not None and cache_key is not None:
                cache[cache_key] = value
                try:
                    cache.move_to_end(cache_key)
                except Exception:
                    pass
                maxsize = getattr(self, '_preview_cache_maxsize', context['preview_cache_size'])
                try:
                    while len(cache) > int(maxsize):
                        cache.popitem(last=False)
                except Exception:
                    pass
        except Exception:
            pass
        return value

    def _try_cache_preview_json(self, preview_bytes, source_meta, original_type, cache_key, preview_size_local, context, safe_text_decode):
        try:
            if source_meta.get('serialization') == 'json' or original_type in {'dict', 'list', 'tuple', 'set'}:
                txt = safe_text_decode(preview_bytes)
                if txt:
                    import json
                    try:
                        obj = json.loads(txt)
                        pretty = json.dumps(obj, ensure_ascii=False, indent=2)
                        snippet = pretty[:preview_size_local]
                        if len(pretty) > preview_size_local:
                            snippet += '...'
                        return self._cache_preview_result(snippet, cache_key, preview_size_local, context)
                    except Exception:
                        truncated = txt[:preview_size_local]
                        if len(txt) > preview_size_local:
                            truncated += '...'
                        return self._cache_preview_result(truncated, cache_key, preview_size_local, context)
        except Exception:
            pass
        return None

    def _extract_semantic_metadata_for_chunk(self, context, chunk_idx):
        registry = context['registry']
        dataset_id = context['dataset_id']
        semantic_metadata_cache = context['semantic_metadata_cache']
        per_chunk_metadata_dir = context['per_chunk_metadata_dir']

        if registry is None or dataset_id is None:
            return None

        if chunk_idx in semantic_metadata_cache:
            return semantic_metadata_cache[chunk_idx]

        try:
            import load as load_module
        except Exception as exc:
            logging.warning(f"Unable to import load module for semantic metadata: {exc}")
            semantic_metadata_cache[chunk_idx] = None
            return None

        element_file = os.path.join(per_chunk_metadata_dir, f"element_records_{dataset_id}_chunk_{chunk_idx}.json")
        try:
            chunk_meta = load_module.get_chunk_metadata(registry, dataset_id, chunk_idx)
            start_pos = int(chunk_meta.get('start_position', chunk_meta.get('start', 0) or 0))
            length_val = chunk_meta.get('length')
            if length_val is None:
                length_val = chunk_meta.get('length_bytes', 0)
            length_val = int(length_val or 0)
            end_pos = start_pos + length_val
            elements_list = list(load_module.get_elements_in_range(registry, dataset_id, start_pos, end_pos))
            preview = self._generate_content_preview(elements_list, load_module, context, preview_size_local=10 ** 7)
            try:
                element_file = element_file + ".gz"
                import gzip
                with gzip.open(element_file, "wt", encoding="utf-8") as f:
                    import json as _json
                    _json.dump({
                        "dataset_id": dataset_id,
                        "chunk_index": chunk_idx,
                        "start_position": start_pos,
                        "length": length_val,
                        "element_records": elements_list,
                        "content_preview": preview,
                        "byte_count": len(elements_list),
                        "element_count": len(elements_list)
                    }, f, ensure_ascii=False, indent=2)
                for elem in elements_list:
                    sig = None
                    if isinstance(elem, dict):
                        for key in ("element_signature", "signature", "cid", "id"):
                            if key in elem and elem.get(key):
                                sig = str(elem.get(key))
                                break
                        if sig is None:
                            try:
                                eidx = int(elem.get('element_index')) if 'element_index' in elem else None
                            except Exception:
                                eidx = None
                            sig = f"{dataset_id}:chunk_{chunk_idx}:elem_{eidx if eidx is not None else 'unknown'}"
                        context['cid_index'][sig] = {
                            'element_records_file': element_file,
                            'chunk_index': chunk_idx,
                            'element_index': elem.get('element_index')
                        }
            except Exception as exc:
                logging.warning(f"Failed to write per-chunk metadata file {element_file}: {exc}")
            metadata = {
                'chunk_index': chunk_idx,
                'start_position': start_pos,
                'length': length_val,
                'element_records_file': element_file,
                'byte_count': len(elements_list),
                'element_count': len(elements_list)
            }
        except Exception as exc:
            logging.warning(f"Failed to extract semantic metadata for chunk {chunk_idx}: {exc}")
            metadata = {
                'chunk_index': chunk_idx,
                'error': str(exc),
                'element_records_file': element_file,
                'byte_count': 0,
                'element_count': 0
            }

        semantic_metadata_cache[chunk_idx] = metadata
        return metadata

    def _attach_semantic_metadata(self, context, connection_dict, source_matrix_idx, target_matrix_idx):
        if not context['include_element_metadata']:
            return
        registry = context['registry']
        dataset_id = context['dataset_id']
        matrix_to_chunk_map = context['matrix_to_chunk_map']
        if registry is None or dataset_id is None or not matrix_to_chunk_map:
            return

        try:
            if source_matrix_idx in matrix_to_chunk_map:
                src_chunk_idx = matrix_to_chunk_map[source_matrix_idx]
                src_meta = self._extract_semantic_metadata_for_chunk(context, int(src_chunk_idx))
                if src_meta is not None:
                    connection_dict['source_metadata'] = src_meta
            if target_matrix_idx in matrix_to_chunk_map:
                tgt_chunk_idx = matrix_to_chunk_map[target_matrix_idx]
                tgt_meta = self._extract_semantic_metadata_for_chunk(context, int(tgt_chunk_idx))
                if tgt_meta is not None:
                    connection_dict['target_metadata'] = tgt_meta
        except Exception as exc:
            logging.warning(f"Failed to attach semantic metadata: {exc}")

    def _validate_matrices(self):
        valid_matrices = []
        valid_indices = []
        valid_matrix_types = []

        if not hasattr(self, 'matrices'):
            logging.warning("No matrices attribute found, creating empty list")
            self.matrices = []
            return valid_matrices, valid_indices, valid_matrix_types

        if not isinstance(self.matrices, list):
            logging.warning("matrices attribute is not a list, returning empty connections")
            return valid_matrices, valid_indices, valid_matrix_types

        for i, matrix in enumerate(self.matrices):
            try:
                if matrix is None:
                    continue

                if isinstance(matrix, torch.Tensor):
                    matrix_np = matrix.detach().cpu().numpy()
                elif isinstance(matrix, np.ndarray):
                    matrix_np = matrix
                elif hasattr(matrix, 'toarray'):
                    matrix_np = matrix.toarray()
                elif hasattr(matrix, 'todense'):
                    matrix_np = matrix.todense()
                else:
                    continue

                if matrix_np.size == 0:
                    continue

                if np.any(np.isnan(matrix_np)) or np.any(np.isinf(matrix_np)):
                    continue

                valid_matrices.append(matrix_np)
                try:
                    mtype = self._detect_matrix_type(matrix_np) if hasattr(self, '_detect_matrix_type') else None
                    if isinstance(mtype, str):
                        try:
                            mtype_enum = MatrixType[mtype.upper()]
                        except Exception:
                            mtype_enum = MatrixType.GENERAL
                    elif isinstance(mtype, MatrixType):
                        mtype_enum = mtype
                    else:
                        mtype_enum = MatrixType.GENERAL
                except Exception:
                    mtype_enum = MatrixType.GENERAL
                valid_matrix_types.append(mtype_enum)
                valid_indices.append(i)
            except Exception as exc:
                logging.warning(f"Skipping invalid matrix at index {i}: {exc}")
                continue

        return valid_matrices, valid_indices, valid_matrix_types

    def _generate_coordinates(self, valid_matrices, valid_indices):
        coords3d = []
        coord_failures = 0
        for i, matrix_np in enumerate(valid_matrices):
            try:
                coords = self._generate_matrix_coordinates_safe(matrix_np, valid_indices[i])
                coords3d.append(coords)
            except Exception as exc:
                logging.warning(f"Failed to generate coordinates for matrix {valid_indices[i]}: {exc}")
                coord_failures += 1
                coords3d.append(np.array([0.5, 0.5, 0.5]))

        coords3d = np.array(coords3d)
        logging.info(f"Coordinate generation complete: {len(coords3d) - coord_failures} successful, {coord_failures} defaults")
        return coords3d

    def _extract_features(self, valid_matrices, valid_indices, valid_matrix_types, num_dims):
        logging.info(f"Starting feature extraction for {len(valid_matrices)} matrices using batch size {500}...")
        features = []
        batch_size = 500
        total_batches = (len(valid_matrices) + batch_size - 1) // batch_size

        for batch_idx, batch_features in enumerate(
            self._batch_project_and_extract_features(
                valid_matrices,
                valid_indices,
                num_dims,
                batch_size,
                valid_matrix_types=valid_matrix_types,
            )
        ):
            features.extend(batch_features)
            logging.debug(f"Extracted features for batch {batch_idx + 1}/{total_batches}, batch size: {len(batch_features)}")
            if (batch_idx + 1) * batch_size % 5000 == 0:
                logging.info(
                    f"Feature extraction progress: {min((batch_idx + 1) * batch_size, len(valid_matrices))} / {len(valid_matrices)} matrices"
                )

        logging.info(f"Processing {len(features)} feature vectors into {num_dims}D normalized array...")
        try:
            features = np.array(features, dtype=np.float64)
            logging.debug(f"Feature matrix shape: {features.shape}, dtype: {features.dtype}")
            logging.info(f"Using raw features without PCA to preserve diversity, shape: {features.shape}")
            norms = np.linalg.norm(features, axis=1)
            zero_norm_count = np.sum(norms < 1e-10)
            logging.debug(
                f"Feature norms - min: {norms.min():.6f}, max: {norms.max():.6f}, zero norms: {zero_norm_count}"
            )
            if np.all(norms < 1e-10):
                logging.warning("All feature vectors have zero norm, returning empty connections")
                return None
            norms = norms[:, np.newaxis] + 1e-10
            features = features / norms * 7.0
            logging.info(f"Feature normalization complete, final shape: {features.shape}, radius: 7.0")
            return features
        except Exception as exc:
            logging.error(f"Failed to process features: {exc}")
            return None

    def _prepare_projections(self, valid_matrices, features):
        logging.info(f"Projecting {len(valid_matrices)} matrices to hypersphere with variance preservation...")
        projected_points = []
        projection_norms = []
        projection_failures = 0

        try:
            if hasattr(self, '_projection_distances') and len(self._projection_distances) == len(valid_matrices):
                proj_dists = list(self._projection_distances)
                logging.debug("Using existing projection distances")
            elif hasattr(self.__class__, '_projection_distances_global') and len(self.__class__._projection_distances_global) == len(valid_matrices):
                proj_dists = list(self.__class__._projection_distances_global)
                logging.debug("Using global projection distances")
            else:
                proj_dists = None
                logging.debug("No projection distances available, will compute")
        except Exception:
            proj_dists = None

        for matrix_np in valid_matrices:
            try:
                flat_original = np.asarray(matrix_np, dtype=float).ravel()
                original_norm = float(np.linalg.norm(flat_original))
                if original_norm > 1e-10:
                    point = (flat_original / original_norm) * 7.0
                else:
                    point = np.ones_like(flat_original)
                    point = (point / (np.linalg.norm(point) + 1e-12)) * 7.0
                projected_points.append(point)
                projection_norms.append(7.0)
            except Exception as exc:
                logging.warning(f"Hypersphere projection failed for matrix: {exc}")
                projection_failures += 1
                projected_points.append(np.array([1.0]))
                projection_norms.append(1.0)

        logging.info(
            f"Hypersphere projection complete: {len(projected_points) - projection_failures} successful, {projection_failures} failures"
        )

        projection_norms = np.array(projection_norms)
        norm_mean = np.mean(projection_norms)
        norm_std = np.std(projection_norms)
        logging.info(
            f"Projection norm statistics - mean: {norm_mean:.6f}, std: {norm_std:.6f}, min: {projection_norms.min():.6f}, max: {projection_norms.max():.6f}"
        )
        self._projection_norms = projection_norms.tolist()
        self._projection_norm_mean = float(norm_mean)
        self._projection_norm_std = float(norm_std)

        try:
            lengths = [len(x) for x in projected_points]
            max_len = max(lengths) if lengths else 0
            if max_len > 0:
                radius_target = 7.0
                for idx, point in enumerate(projected_points):
                    if len(point) < max_len:
                        padded = np.zeros(max_len, dtype=float)
                        padded[:len(point)] = point
                        norm = np.linalg.norm(padded)
                        if norm > 1e-12:
                            padded = (padded / norm) * radius_target
                        else:
                            padded = padded + (1.0 / np.sqrt(max_len))
                            padded = (padded / np.linalg.norm(padded)) * radius_target
                        projected_points[idx] = padded
                    else:
                        point = np.asarray(point, dtype=float)
                        norm = np.linalg.norm(point)
                        if norm > 1e-12:
                            projected_points[idx] = (point / norm) * radius_target
                        else:
                            projected_points[idx] = np.ones(max_len, dtype=float) * (radius_target / np.sqrt(max_len))
        except Exception:
            pass

        proj_dists = self._compute_projection_distances(projected_points, proj_dists)

        try:
            self._projection_distances = [
                float(d) if (isinstance(d, (int, float)) and not np.isnan(d) and not np.isinf(d)) else 0.0 for d in proj_dists
            ]
        except Exception:
            try:
                self._projection_distances = [float(d) if d is not None else 0.0 for d in proj_dists]
            except Exception:
                self._projection_distances = []

        logging.info(f"Projection distances computed: {len(proj_dists)} distances for {len(projected_points)} points")
        logging.debug(f"proj_dists sample (first 5): {proj_dists[:5] if len(proj_dists) > 0 else 'empty'}")

        try:
            self._projected_points = []
            for point in projected_points:
                try:
                    arr = np.asarray(point, dtype=float)
                    self._projected_points.append(arr.flatten().tolist())
                except Exception:
                    self._projected_points.append([float(v) for v in list(point)])
        except Exception:
            self._projected_points = []

        return np.asarray(projected_points, dtype=float), proj_dists, projection_norms

    def _compute_projection_distances(self, projected_points, proj_dists):
        if proj_dists is not None and len(proj_dists) == len(projected_points):
            return proj_dists

        proj_dists = []
        try:
            arr_pts = np.asarray(projected_points, dtype=float)
            if arr_pts.ndim == 1:
                x0 = np.ones_like(arr_pts)
                x0 = x0 / (np.linalg.norm(x0) + 1e-12)
                proj_dists = [float(self.local_distance_sphere(x0, arr_pts))]
            else:
                x0 = np.ones(arr_pts.shape[1], dtype=float)
                x0 = x0 / (np.linalg.norm(x0) + 1e-12)
                try:
                    dists = self.local_distance_sphere(x0, arr_pts)
                    proj_dists = [float(d) for d in np.asarray(dists)]
                except Exception:
                    proj_dists = []
                    for point in projected_points:
                        try:
                            base = np.ones_like(point)
                            base = base / (np.linalg.norm(base) + 1e-12)
                            proj_dists.append(float(self.local_distance_sphere(base, point)))
                        except Exception:
                            proj_dists.append(0.0)
        except Exception:
            proj_dists = []
            for point in projected_points:
                try:
                    base = np.ones_like(point)
                    base = base / (np.linalg.norm(base) + 1e-12)
                    proj_dists.append(float(self.local_distance_sphere(base, point)))
                except Exception:
                    proj_dists.append(0.0)

        return proj_dists

    def _run_connection_batches(
        self,
        features,
        projected_points,
        coords3d,
        valid_indices,
        context,
        batch_size_conn,
        use_memmap,
        memmap_dir,
        use_ann,
        ann_k,
        block_size,
        candidate_k,
        projection_norms,
        proj_dists,
    ):
        logging.info(
            f"Starting connection search with batch_size={batch_size_conn or min(100, len(valid_indices))}, "
            f"use_memmap={use_memmap}, use_ann={use_ann}"
        )

        connections = {}
        if batch_size_conn is None:
            batch_size_conn = min(100, len(valid_indices))

        matrix_to_chunk_map = context['matrix_to_chunk_map']
        per_chunk_full_connections = context['per_chunk_full_connections']
        cid_index = context['cid_index']
        per_chunk_metadata_dir = context['per_chunk_metadata_dir']
        dataset_id = context['dataset_id']
        min_similarity = context['min_similarity']
        min_ratio = context['min_ratio']
        top_k = context['top_k']
        connection_limit = context['connection_limit']

        memmap_tmpdir = None
        features_mmap = None
        proj_mmap = None
        coords_mmap = None

        try:
            N = len(valid_indices)
            logging.debug(f"Connection search setup: N={N} matrices, memmap threshold=20000")
            if use_memmap or N > 20000:
                import shutil

                memmap_tmpdir = memmap_dir or tempfile.mkdtemp(prefix='tp_memmap_')
                features_path = os.path.join(memmap_tmpdir, 'features.dat')
                proj_path = os.path.join(memmap_tmpdir, 'projected_points.dat')
                coords_path = os.path.join(memmap_tmpdir, 'coords3d.dat')

                fshape = features.shape
                features_mmap = np.memmap(features_path, dtype=np.float32, mode='w+', shape=fshape)
                features_mmap[:] = features.astype(np.float32)

                proj_arr = np.asarray(projected_points, dtype=np.float32)
                proj_mmap = np.memmap(proj_path, dtype=np.float32, mode='w+', shape=proj_arr.shape)
                proj_mmap[:] = proj_arr

                coords_arr = np.asarray(coords3d, dtype=np.float32)
                coords_mmap = np.memmap(coords_path, dtype=np.float32, mode='w+', shape=coords_arr.shape)
                coords_mmap[:] = coords_arr

                del features
                del projected_points
                del coords3d
            else:
                features_mmap = features
                proj_mmap = np.asarray(projected_points, dtype=float)
                coords_mmap = np.asarray(coords3d, dtype=float)

            nbrs = None
            if use_ann:
                try:
                    logging.info(f"Building ANN index with {ann_k} neighbors using cosine metric...")
                    nbrs = NearestNeighbors(
                        n_neighbors=min(ann_k, features_mmap.shape[0]),
                        metric='cosine',
                        algorithm='auto'
                    )
                    nbrs.fit(features_mmap)
                    logging.info(
                        f"ANN index built successfully for {features_mmap.shape[0]} vectors"
                    )
                except Exception as exc:
                    logging.warning(f"ANN index build failed: {exc}; falling back to block dot streaming")
                    nbrs = None

            total_conn_batches = (len(valid_indices) + batch_size_conn - 1) // batch_size_conn
            logging.info(f"Processing {total_conn_batches} connection batches of size {batch_size_conn}...")

            for i in range(0, len(valid_indices), batch_size_conn):
                batch_end = min(i + batch_size_conn, len(valid_indices))
                batch_num = i // batch_size_conn + 1
                logging.debug(f"Processing connection batch {batch_num}/{total_conn_batches}, indices {i}:{batch_end}")

                batch_features = np.asarray(features_mmap[i:batch_end], dtype=float)
                batch_indices = valid_indices[i:batch_end]

                try:
                    bf = np.asarray(batch_features, dtype=float)
                    B = bf.shape[0]
                    candidate_idx = None
                    candidate_sims = None

                    if nbrs is not None:
                        try:
                            logging.debug(f"Using ANN to find {ann_k} neighbors for batch of {B} sources")
                            dists, inds = nbrs.kneighbors(bf, return_distance=True)
                            sims = 1.0 - dists
                            candidate_idx = inds
                            candidate_sims = sims
                            logging.debug(
                                f"ANN found candidates with similarity range [{sims.min():.3f}, {sims.max():.3f}]"
                            )
                        except Exception as exc:
                            logging.warning(
                                f"ANN kneighbors failed for batch starting at {i}: {exc}; falling back to block streaming"
                            )
                            candidate_idx = None
                            candidate_sims = None

                    if candidate_idx is None:
                        N_total = features_mmap.shape[0]
                        k = min(candidate_k or 256, N_total)
                        logging.debug(
                            f"Using block streaming with block_size={block_size}, keeping top-{k} per source"
                        )
                        topk_sims = np.full((B, k), -np.inf, dtype=float)
                        topk_idx = np.full((B, k), -1, dtype=int)
                        bf64 = bf.astype(np.float64)

                        for j in range(0, N_total, block_size):
                            F_block = features_mmap[j:j + block_size]
                            if F_block.size == 0:
                                continue
                            Fb = np.asarray(F_block, dtype=float)
                            sims_block = np.dot(bf64, Fb.T)
                            M = sims_block.shape[1]
                            block_indices = np.arange(j, j + M)
                            concat_sims = np.concatenate([topk_sims, sims_block], axis=1)
                            concat_idx = np.concatenate(
                                [topk_idx, np.broadcast_to(block_indices, (B, M))],
                                axis=1
                            )
                            part = np.argpartition(concat_sims, -k, axis=1)[:, -k:]
                            rows = np.arange(B)[:, None]
                            topk_sims = concat_sims[rows, part]
                            topk_idx = concat_idx[rows, part]
                            order = np.argsort(-topk_sims, axis=1)
                            topk_sims = topk_sims[rows, order]
                            topk_idx = topk_idx[rows, order]

                        candidate_idx = topk_idx
                        candidate_sims = topk_sims

                except Exception as exc:
                    logging.warning(
                        f"Failed to compute similarities/candidates for batch starting at {i}: {exc}"
                    )
                    candidate_idx = np.zeros((len(batch_features), 0), dtype=int)
                    candidate_sims = np.zeros((len(batch_features), 0), dtype=float)

                for batch_idx, src_idx in enumerate(batch_indices):
                    targets = []
                    try:
                        idx_row = (
                            candidate_idx[batch_idx]
                            if candidate_idx is not None and candidate_idx.size > 0
                            else np.array([], dtype=int)
                        )
                        sims_row = (
                            candidate_sims[batch_idx]
                            if candidate_sims is not None and candidate_sims.size > 0
                            else np.array([], dtype=float)
                        )
                        if sims_row.size > 0:
                            significant_pos = np.where(sims_row > float(min_similarity))[0]
                            significant_indices = [int(idx_row[pos]) for pos in significant_pos]
                            logging.debug(
                                f"Source {src_idx}: found {len(significant_indices)} candidates above similarity threshold {min_similarity}"
                            )
                        else:
                            significant_indices = []
                            logging.debug(f"Source {src_idx}: no candidates found (empty similarity row)")
                    except Exception as inner_exc:
                        logging.warning(
                            f"Could not process candidate similarities for index {src_idx}: {inner_exc}"
                        )
                        significant_indices = []

                    sig_filtered = [
                        t for t in significant_indices if t != batch_idx + i and t < len(valid_indices)
                    ]

                    if sig_filtered:
                        src_local = i + batch_idx
                        x_i = proj_mmap[src_local]
                        x_js = [proj_mmap[t] for t in sig_filtered]
                        try:
                            x_js_arr = np.asarray(x_js, dtype=float)
                        except Exception:
                            x_js_arr = None
                    else:
                        x_js_arr = None

                    if x_js_arr is None or (isinstance(x_js_arr, np.ndarray) and x_js_arr.size == 0):
                        sig_filtered = []

                    if sig_filtered:
                        try:
                            v_ij_batch = self.log_map_sphere(x_i, x_js_arr)
                            v_ji_batch = self.log_map_sphere(x_js_arr, np.broadcast_to(x_i, v_ij_batch.shape))
                            v_ij_t_batch = self.parallel_transport_sphere(x_i, x_js_arr, v_ij_batch)
                            vnorm_batch = np.linalg.norm(v_ij_batch, axis=1)
                            vnorm_j_batch = np.linalg.norm(v_ji_batch, axis=1)
                        except Exception:
                            v_ij_batch = None
                            v_ji_batch = None
                            v_ij_t_batch = None
                            vnorm_batch = None
                            vnorm_j_batch = None
                    else:
                        v_ij_batch = None
                        v_ji_batch = None
                        v_ij_t_batch = None
                        vnorm_batch = None
                        vnorm_j_batch = None

                    for k_idx, tgt_idx in enumerate(sig_filtered):
                        try:
                            if tgt_idx >= len(valid_indices):
                                continue

                            try:
                                x_i_proj = proj_mmap[i + batch_idx]
                                x_j_proj = proj_mmap[tgt_idx]
                                phys_dist = float(self.local_distance_sphere(x_i_proj, x_j_proj))
                            except Exception as exc:
                                logging.debug(f"Failed spherical distance, using fallback: {exc}")
                                phys_dist = np.linalg.norm(coords_mmap[i + batch_idx] - coords_mmap[tgt_idx])

                            similarity_val = 0.0
                            try:
                                if (
                                    x_js_arr is not None
                                    and candidate_sims is not None
                                    and candidate_sims.shape[0] > batch_idx
                                ):
                                    similarity_val = float(candidate_sims[batch_idx, k_idx])
                                elif candidate_idx is not None and candidate_idx.shape[0] > batch_idx:
                                    pos = np.where(candidate_idx[batch_idx] == tgt_idx)[0]
                                    if pos.size > 0:
                                        similarity_val = float(candidate_sims[batch_idx, pos[0]])
                                    else:
                                        similarity_val = float(
                                            np.dot(features_mmap[i + batch_idx], features_mmap[tgt_idx])
                                        )
                                else:
                                    similarity_val = float(
                                        np.dot(features_mmap[i + batch_idx], features_mmap[tgt_idx])
                                    )
                            except Exception:
                                similarity_val = 0.0

                            try:
                                src_feat = np.asarray(features_mmap[i + batch_idx], dtype=float)
                                tgt_feat = np.asarray(features_mmap[tgt_idx], dtype=float)
                                hd_dist = float(np.linalg.norm(src_feat - tgt_feat))
                            except Exception:
                                hd_dist = float(
                                    np.sqrt(max(0.0, 2 * (1 - np.clip(similarity_val, -1, 1))))
                                )

                            if x_js_arr is not None and v_ij_batch is not None and vnorm_batch is not None:
                                v_ij = v_ij_batch[k_idx]
                                vnorm = float(vnorm_batch[k_idx])
                            else:
                                try:
                                    x_j = proj_mmap[tgt_idx]
                                    v_ij = self.log_map_sphere(x_i, x_j)
                                    vnorm = float(np.linalg.norm(v_ij))
                                except Exception:
                                    v_ij = np.array([])
                                    vnorm = 0.0

                            phys_dist = vnorm
                            ratio = float('inf') if hd_dist == 0.0 else phys_dist / hd_dist
                            accept_connection = (
                                ratio > float(min_ratio)
                                or (
                                    phys_dist < 0.1
                                    and hd_dist < 0.1
                                    and similarity_val > float(min_similarity)
                                )
                            )

                            if not accept_connection:
                                continue

                            try:
                                src_local = i + batch_idx
                                tgt_local = tgt_idx
                                local_energy = float(proj_dists[src_local])
                            except Exception:
                                local_energy = 0.0

                            try:
                                target_energy = float(proj_dists[tgt_local])
                            except Exception:
                                target_energy = 0.0

                            try:
                                if (
                                    x_js_arr is not None
                                    and v_ji_batch is not None
                                    and v_ij_t_batch is not None
                                    and vnorm_j_batch is not None
                                ):
                                    v_ji = v_ji_batch[k_idx]
                                    v_ij_t = v_ij_t_batch[k_idx]
                                    vnorm_j = float(vnorm_j_batch[k_idx])
                                else:
                                    v_ji = self.log_map_sphere(x_j, x_i)
                                    v_ij_t = self.parallel_transport_sphere(x_i, x_j, v_ij)
                                    vnorm_j = float(np.linalg.norm(v_ji)) if v_ji is not None else 0.0
                            except Exception:
                                v_ji = np.array([])
                                v_ij_t = np.array([])
                                vnorm_j = 0.0

                            try:
                                if vnorm > 1e-12 and vnorm_j > 1e-12:
                                    denom = vnorm * vnorm_j
                                    dotp = np.clip(np.sum(v_ij_t * v_ji) / denom, -1.0, 1.0)
                                    reciprocal_angle = float(math.acos(dotp))
                                else:
                                    reciprocal_angle = 0.0
                            except Exception:
                                reciprocal_angle = 0.0

                            try:
                                sphere_radius = 7.0
                                local_curvature = float(-1.0 / (sphere_radius ** 2))
                            except Exception:
                                local_curvature = 0.0

                            try:
                                energy_gradient = float(target_energy - local_energy)
                            except Exception:
                                energy_gradient = 0.0

                            try:
                                geodesic_error = float(abs(vnorm - phys_dist))
                            except Exception:
                                geodesic_error = 0.0

                            try:
                                src_projection_norm = float(projection_norms[i + batch_idx])
                                tgt_projection_norm = float(projection_norms[tgt_idx])
                                norm_variance = float(abs(src_projection_norm - tgt_projection_norm))
                                norm_mean = self._projection_norm_mean
                                norm_variance_relative = float(norm_variance / (norm_mean + 1e-10))
                            except Exception:
                                src_projection_norm = 1.0
                                tgt_projection_norm = 1.0
                                norm_variance = 0.0
                                norm_variance_relative = 0.0

                            full_entry = {
                                "source_idx": src_idx,
                                "target_idx": valid_indices[tgt_idx],
                                "high_dim_dist": float(hd_dist),
                                "hyperdimensional_dist": float(hd_dist),
                                "physical_dist": float(phys_dist),
                                "ratio": float(ratio),
                                "strength": float(similarity_val),
                                "dimensions": np.argsort(
                                    np.abs(
                                        np.asarray(features_mmap[i + batch_idx], dtype=float)
                                        - np.asarray(features_mmap[tgt_idx], dtype=float)
                                    )
                                )[-3:].tolist(),
                                "log_map": v_ij.tolist() if hasattr(v_ij, 'tolist') else [],
                                "log_map_norm": float(vnorm),
                                "transported_log_map": v_ij_t.tolist() if hasattr(v_ij_t, 'tolist') else [],
                                "reciprocal_angle": float(reciprocal_angle),
                                "local_curvature": float(local_curvature),
                                "local_energy": float(local_energy),
                                "target_energy": float(target_energy),
                                "energy_gradient": float(energy_gradient),
                                "geodesic_error": float(geodesic_error),
                                "source_projection_norm": float(src_projection_norm),
                                "target_projection_norm": float(tgt_projection_norm),
                                "norm_variance": float(norm_variance),
                                "norm_variance_relative": float(norm_variance_relative),
                            }

                            summary_entry = {
                                "source_idx": src_idx,
                                "target_idx": valid_indices[tgt_idx],
                                "high_dim_dist": float(hd_dist),
                                "hyperdimensional_dist": float(hd_dist),
                                "physical_dist": float(phys_dist),
                                "ratio": float(ratio),
                                "strength": float(similarity_val),
                                "dimensions": np.argsort(
                                    np.abs(
                                        np.asarray(features_mmap[i + batch_idx], dtype=float)
                                        - np.asarray(features_mmap[tgt_idx], dtype=float)
                                    )
                                )[-3:].tolist(),
                                "log_map_norm": float(vnorm),
                                "reciprocal_angle": float(reciprocal_angle),
                                "local_curvature": float(local_curvature),
                                "local_energy": float(local_energy),
                                "target_energy": float(target_energy),
                                "energy_gradient": float(energy_gradient),
                                "geodesic_error": float(geodesic_error),
                                "source_projection_norm": float(src_projection_norm),
                                "target_projection_norm": float(tgt_projection_norm),
                                "norm_variance": float(norm_variance),
                                "norm_variance_relative": float(norm_variance_relative),
                            }

                            self._attach_semantic_metadata(
                                context,
                                summary_entry,
                                src_idx,
                                valid_indices[tgt_idx],
                            )
                            targets.append(summary_entry)

                            try:
                                src_chunk = matrix_to_chunk_map.get(src_idx)
                                if src_chunk is not None:
                                    per_chunk_full_connections[int(src_chunk)].append(full_entry)
                            except Exception:
                                pass
                        except Exception as exc:
                            logging.warning(
                                f"Could not process connection from {src_idx} to {valid_indices[tgt_idx] if tgt_idx < len(valid_indices) else tgt_idx}: {exc}"
                            )
                            continue

                    if targets:
                        connections[src_idx] = sorted(
                            targets,
                            key=lambda x: x["strength"],
                            reverse=True,
                        )[:connection_limit]
                    else:
                        connections[src_idx] = self._fallback_topk_connections(
                            context,
                            candidate_idx,
                            candidate_sims,
                            batch_idx,
                            batch_features,
                            features_mmap,
                            proj_mmap,
                            coords_mmap,
                            proj_dists,
                            projection_norms,
                            valid_indices,
                            i,
                            top_k,
                        )

            self.hyperdimensional_connections = connections

            total_edges = sum(len(targets) for targets in connections.values())
            non_empty_sources = len([s for s in connections.values() if len(s) > 0])
            edge_distribution = [len(targets) for targets in connections.values()]

            logging.info("Connection discovery complete:")
            logging.info(f"  - Found hyperdimensional connections for {len(connections)} matrices")
            logging.info(f"  - Total edges: {total_edges}")
            logging.info(f"  - Non-empty sources: {non_empty_sources}/{len(connections)}")
            if edge_distribution:
                logging.info(
                    f"  - Edges per source - min: {min(edge_distribution)}, max: {max(edge_distribution)}, "
                    f"mean: {np.mean(edge_distribution):.2f}"
                )

            try:
                all_indices = set(connections.keys())
                for targets_list in list(connections.values()):
                    if not isinstance(targets_list, list):
                        continue
                    for target_entry in targets_list:
                        tidx = target_entry.get('target_idx') if isinstance(target_entry, dict) else None
                        if tidx is None:
                            continue
                        try:
                            tidx = int(tidx)
                        except Exception:
                            pass
                        all_indices.add(tidx)
                for idx in all_indices:
                    connections.setdefault(idx, [])
            except Exception:
                pass

            self._persist_connection_artifacts(
                connections,
                per_chunk_full_connections,
                cid_index,
                per_chunk_metadata_dir,
                dataset_id,
            )

            return connections

        finally:
            try:
                if memmap_tmpdir:
                    logging.debug(f"Cleaning up temporary memmap directory: {memmap_tmpdir}")

                    def _safe_close_memmap(mm):
                        try:
                            if mm is None:
                                return
                            if isinstance(mm, np.memmap):
                                try:
                                    mm.flush()
                                except Exception:
                                    pass
                                try:
                                    if hasattr(mm, '_mmap') and mm._mmap:
                                        mm._mmap.close()
                                except Exception:
                                    pass
                            elif hasattr(mm, 'flush'):
                                try:
                                    mm.flush()
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    try:
                        _safe_close_memmap(features_mmap)
                    except Exception:
                        pass
                    try:
                        _safe_close_memmap(proj_mmap)
                    except Exception:
                        pass
                    try:
                        _safe_close_memmap(coords_mmap)
                    except Exception:
                        pass

                    try:
                        import shutil

                        shutil.rmtree(memmap_tmpdir)
                        logging.debug("Successfully cleaned up memmap directory")
                    except Exception as exc:
                        logging.warning(f"Failed to clean up memmap directory: {exc}")
            except NameError:
                pass

    def _fallback_topk_connections(
        self,
        context,
        candidate_idx,
        candidate_sims,
        batch_idx,
        batch_features,
        features_mmap,
        proj_mmap,
        coords_mmap,
        proj_dists,
        projection_norms,
        valid_indices,
        batch_start,
        top_k,
    ):
        min_similarity = context['min_similarity']
        matrix_to_chunk_map = context['matrix_to_chunk_map']
        per_chunk_full_connections = context['per_chunk_full_connections']

        if not top_k or not isinstance(top_k, int) or top_k <= 0:
            return []

        try:
            if candidate_idx is not None and candidate_idx.shape[1] > 0:
                sim_values = candidate_sims[batch_idx].copy()
                idxs = candidate_idx[batch_idx]
                mask_self = idxs == (batch_start + batch_idx)
                if mask_self.any():
                    sim_values[mask_self] = -np.inf
                order = np.argsort(sim_values)[-min(len(sim_values), top_k):][::-1]
                top_idxs = idxs[order].tolist()
            else:
                bf_row = np.asarray(batch_features[batch_idx], dtype=float)
                full_sims = np.dot(bf_row, features_mmap.T)
                if batch_start + batch_idx < full_sims.size:
                    full_sims[batch_start + batch_idx] = -np.inf
                top_idxs = np.argsort(full_sims)[-top_k:][::-1].tolist()
        except Exception:
            top_idxs = []

        fallback_targets = []
        top_filtered = [
            t for t in top_idxs if t < len(valid_indices) and t != batch_start + batch_idx
        ]

        if top_filtered:
            x_i = proj_mmap[batch_start + batch_idx]
            try:
                x_js_top = np.asarray([proj_mmap[t] for t in top_filtered], dtype=float)
                v_ij_top = self.log_map_sphere(x_i, x_js_top)
            except Exception:
                x_js_top = None
                v_ij_top = None
        else:
            x_js_top = None
            v_ij_top = None

        for k_idx, tgt_idx in enumerate(top_filtered if top_filtered else top_idxs):
            try:
                if tgt_idx >= len(valid_indices):
                    continue

                try:
                    x_i_proj = proj_mmap[batch_start + batch_idx]
                    x_j_proj = proj_mmap[tgt_idx]
                    phys_dist = float(self.local_distance_sphere(x_i_proj, x_j_proj))
                except Exception:
                    phys_dist = np.linalg.norm(
                        coords_mmap[batch_start + batch_idx] - coords_mmap[tgt_idx]
                    )

                try:
                    if candidate_sims is not None and candidate_sims.shape[0] > batch_idx:
                        if candidate_idx is not None and candidate_idx.shape[0] > batch_idx:
                            pos = np.where(candidate_idx[batch_idx] == tgt_idx)[0]
                            if pos.size > 0:
                                similarity_val = float(candidate_sims[batch_idx, pos[0]])
                            else:
                                similarity_val = float(
                                    np.dot(
                                        np.asarray(features_mmap[batch_start + batch_idx], dtype=float),
                                        np.asarray(features_mmap[tgt_idx], dtype=float),
                                    )
                                )
                        else:
                            similarity_val = float(
                                np.dot(
                                    np.asarray(features_mmap[batch_start + batch_idx], dtype=float),
                                    np.asarray(features_mmap[tgt_idx], dtype=float),
                                )
                            )
                    else:
                        similarity_val = float(
                            np.dot(
                                np.asarray(features_mmap[batch_start + batch_idx], dtype=float),
                                np.asarray(features_mmap[tgt_idx], dtype=float),
                            )
                        )
                except Exception:
                    similarity_val = 0.0

                try:
                    src_feat = np.asarray(features_mmap[batch_start + batch_idx], dtype=float)
                    tgt_feat = np.asarray(features_mmap[tgt_idx], dtype=float)
                    hd_dist = float(np.linalg.norm(src_feat - tgt_feat))
                except Exception:
                    hd_dist = float(
                        np.sqrt(max(0.0, 2 * (1 - np.clip(similarity_val, -1, 1))))
                    )

                if x_js_top is not None and v_ij_top is not None and tgt_idx in top_filtered:
                    idx_local = top_filtered.index(tgt_idx)
                    v_ij = v_ij_top[idx_local]
                else:
                    try:
                        v_ij = self.log_map_sphere(
                            proj_mmap[batch_start + batch_idx],
                            proj_mmap[tgt_idx],
                        )
                    except Exception:
                        v_ij = np.array([])

                try:
                    vnorm = float(np.linalg.norm(v_ij)) if v_ij.size else 0.0
                except Exception:
                    vnorm = 0.0

                phys_dist = vnorm
                ratio = float('inf') if hd_dist == 0.0 else phys_dist / hd_dist

                try:
                    local_energy = float(proj_dists[batch_start + batch_idx])
                except Exception:
                    local_energy = 0.0

                try:
                    target_energy = float(proj_dists[tgt_idx])
                except Exception:
                    target_energy = 0.0

                try:
                    v_ij_t_val = self.parallel_transport_sphere(
                        proj_mmap[batch_start + batch_idx],
                        proj_mmap[tgt_idx],
                        v_ij,
                    )
                except Exception:
                    v_ij_t_val = np.array([])

                try:
                    src_projection_norm = float(projection_norms[batch_start + batch_idx])
                    tgt_projection_norm = float(projection_norms[tgt_idx])
                    norm_variance = float(abs(src_projection_norm - tgt_projection_norm))
                    norm_mean = self._projection_norm_mean
                    norm_variance_relative = float(norm_variance / (norm_mean + 1e-10))
                except Exception:
                    src_projection_norm = 1.0
                    tgt_projection_norm = 1.0
                    norm_variance = 0.0
                    norm_variance_relative = 0.0

                full_entry = {
                    "source_idx": valid_indices[batch_start + batch_idx],
                    "target_idx": valid_indices[tgt_idx],
                    "high_dim_dist": float(hd_dist),
                    "hyperdimensional_dist": float(hd_dist),
                    "physical_dist": float(phys_dist),
                    "ratio": float(ratio),
                    "strength": float(similarity_val),
                    "dimensions": [0, 1, 2],
                    "log_map": v_ij.tolist() if hasattr(v_ij, 'tolist') else [],
                    "log_map_norm": float(vnorm),
                    "transported_log_map": v_ij_t_val.tolist() if hasattr(v_ij_t_val, 'tolist') else [],
                    "reciprocal_angle": 0.0,
                    "local_curvature": float(
                        (hd_dist - phys_dist) / (phys_dist + 1e-9)
                    ) if phys_dist != 0 else 0.0,
                    "local_energy": float(local_energy),
                    "target_energy": float(target_energy),
                    "energy_gradient": float(target_energy - local_energy),
                    "geodesic_error": float(abs(vnorm - phys_dist)),
                    "source_projection_norm": float(src_projection_norm),
                    "target_projection_norm": float(tgt_projection_norm),
                    "norm_variance": float(norm_variance),
                    "norm_variance_relative": float(norm_variance_relative),
                }

                summary_entry = {
                    "source_idx": valid_indices[batch_start + batch_idx],
                    "target_idx": valid_indices[tgt_idx],
                    "high_dim_dist": float(hd_dist),
                    "hyperdimensional_dist": float(hd_dist),
                    "physical_dist": float(phys_dist),
                    "ratio": float(ratio),
                    "strength": float(similarity_val),
                    "dimensions": [0, 1, 2],
                    "log_map_norm": float(vnorm),
                    "reciprocal_angle": 0.0,
                    "local_curvature": float(
                        (hd_dist - phys_dist) / (phys_dist + 1e-9)
                    ) if phys_dist != 0 else 0.0,
                    "local_energy": float(local_energy),
                    "target_energy": float(target_energy),
                    "energy_gradient": float(target_energy - local_energy),
                    "geodesic_error": float(abs(vnorm - phys_dist)),
                    "source_projection_norm": float(src_projection_norm),
                    "target_projection_norm": float(tgt_projection_norm),
                    "norm_variance": float(norm_variance),
                    "norm_variance_relative": float(norm_variance_relative),
                }

                self._attach_semantic_metadata(
                    context,
                    summary_entry,
                    valid_indices[batch_start + batch_idx],
                    valid_indices[tgt_idx],
                )
                fallback_targets.append(summary_entry)

                try:
                    src_chunk = matrix_to_chunk_map.get(valid_indices[batch_start + batch_idx])
                    if src_chunk is not None:
                        per_chunk_full_connections[int(src_chunk)].append(full_entry)
                except Exception:
                    pass
            except Exception:
                continue

        return sorted(
            fallback_targets,
            key=lambda x: x.get("strength", 0.0),
            reverse=True,
        )[:context['connection_limit']]

    def _persist_connection_artifacts(
        self,
        connections,
        per_chunk_full_connections,
        cid_index,
        per_chunk_metadata_dir,
        dataset_id,
    ):
        try:
            import concurrent.futures
            import gzip
            import json as _json
            import math

            out_npz_dir = os.path.join(per_chunk_metadata_dir, "npz_connections")
            try:
                os.makedirs(out_npz_dir, exist_ok=True)
            except Exception:
                out_npz_dir = per_chunk_metadata_dir

            def _save_chunk_npz(chunk_idx, entries, compress=True):
                try:
                    fname = os.path.join(
                        out_npz_dir,
                        f"connections_chunk_{dataset_id}_chunk_{int(chunk_idx)}.npz",
                    )
                    srcs = np.array([int(e.get('source_idx', -1)) for e in entries], dtype=np.int64)
                    tgts = np.array([int(e.get('target_idx', -1)) for e in entries], dtype=np.int64)
                    json_strs = np.array([
                        _json.dumps(e, ensure_ascii=False) for e in entries
                    ], dtype=object)
                    if compress:
                        np.savez_compressed(fname, source_idx=srcs, target_idx=tgts, entry_json=json_strs)
                    else:
                        np.savez(fname, source_idx=srcs, target_idx=tgts, entry_json=json_strs)
                    return fname
                except Exception as exc:
                    logging.warning(f"_save_chunk_npz failed for chunk {chunk_idx}: {exc}")
                    return None

            max_workers = min(8, (os.cpu_count() or 1) * 2)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                for chunk_idx, entries in per_chunk_full_connections.items():
                    if not entries:
                        continue
                    futures.append(ex.submit(_save_chunk_npz, chunk_idx, entries, True))
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        pass

            try:
                conn_items = list(connections.items())
                if conn_items:
                    batch_size_save = max(1, min(2048, int(math.sqrt(len(conn_items)) * 4)))
                    batch_dir = os.path.join(out_npz_dir, 'batches')
                    os.makedirs(batch_dir, exist_ok=True)

                    def _save_conn_batch(batch_num, items):
                        try:
                            fname = os.path.join(
                                batch_dir,
                                f"connections_batch_{dataset_id}_{batch_num:06d}.npz",
                            )
                            srcs = np.array([int(k) for k, _ in items], dtype=np.int64)
                            json_lists = np.array([
                                _json.dumps(v, ensure_ascii=False) for _, v in items
                            ], dtype=object)
                            np.savez_compressed(fname, source_idx=srcs, connections_json=json_lists)
                            return fname
                        except Exception as exc:
                            logging.warning(f"_save_conn_batch failed for batch {batch_num}: {exc}")
                            return None

                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 1))) as ex:
                        futures = []
                        for bstart in range(0, len(conn_items), batch_size_save):
                            batch_num = bstart // batch_size_save + 1
                            batch_items = conn_items[bstart:bstart + batch_size_save]
                            futures.append(ex.submit(_save_conn_batch, batch_num, batch_items))
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                future.result()
                            except Exception:
                                pass
            except Exception:
                logging.warning("Failed to batch-save main connections; falling back to JSON.gz")
                try:
                    for chunk_idx, entries in per_chunk_full_connections.items():
                        try:
                            conn_file = os.path.join(
                                per_chunk_metadata_dir,
                                f"connections_chunk_{dataset_id}_chunk_{int(chunk_idx)}.json.gz",
                            )
                            with gzip.open(conn_file, 'wt', encoding='utf-8') as cf:
                                _json.dump(
                                    {
                                        'dataset_id': dataset_id,
                                        'chunk_index': int(chunk_idx),
                                        'connections': entries,
                                    },
                                    cf,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                cid_file = os.path.join(
                    per_chunk_metadata_dir,
                    f"cid_index_{dataset_id}.json.gz",
                )
                with gzip.open(cid_file, 'wt', encoding='utf-8') as cf:
                    _json.dump(cid_index, cf, ensure_ascii=False, indent=2)
            except Exception as exc:
                logging.warning(f"Failed to write CID index file: {exc}")

        except Exception as exc:
            logging.warning(f"Failed to persist connections to disk using .npz pipeline: {exc}")

    def find_hyperdimensional_connections(self, num_dims=8, min_similarity=0.5, min_ratio=5.0, top_k=None,
                                         batch_size_conn=None, use_memmap=False, memmap_dir=None,
                                         use_ann=False, ann_k=128, block_size=1024, candidate_k=256,
                                         registry=None, dataset_id=None, matrix_to_chunk_map=None,
                                         include_element_metadata=True,
                                         preview_size: int = 1024,
                                         preview_cache_size: int = 128):
        """Find connections in hyperdimensional space between matrices and tensors.

        The method supports memory-backed processing and candidate pre-filtering.
        New optional keyword parameters:
            batch_size_conn: int | None - batch size of sources to process sequentially
            use_memmap: bool - whether to persist features/projected_points/coords to disk-backed memmap
            memmap_dir: str | None - directory for memmap temporary files
            use_ann: bool - whether to build an ANN index to pre-filter candidates
            ann_k: int - number of neighbors to retrieve per source from ANN
            block_size: int - block size for block-dot streaming fallback
            candidate_k: int - number of top candidates to keep per source in streaming fallback
            preview_size: int - max bytes used for content preview (default 1024)
            preview_cache_size: int - LRU cache size for previews (per instance)

        The new behavior preserves backward compatibility but allows large datasets to be
        processed without materializing the full NxN similarity matrix.
        """
        logging.info(f"Finding hyperdimensional connections in {num_dims}D space...")

        context = self._prepare_connection_context(
            num_dims,
            min_similarity,
            min_ratio,
            top_k,
            registry,
            dataset_id,
            matrix_to_chunk_map,
            include_element_metadata,
            preview_size,
            preview_cache_size,
        )

        self.hyperdimensional_connections = {}

        valid_matrices, valid_indices, valid_matrix_types = self._validate_matrices()
        total_matrices = len(self.matrices) if isinstance(self.matrices, list) else 0
        logging.info(
            f"Matrix validation complete: {len(valid_matrices)} valid matrices out of {total_matrices} total"
        )

        if not valid_matrices:
            logging.warning("No valid matrices found for hyperdimensional connections")
            return self.hyperdimensional_connections

        if len(valid_matrices) == 1:
            logging.info("Single matrix case - creating entry with empty connections")
            self.hyperdimensional_connections[valid_indices[0]] = []
            return self.hyperdimensional_connections

        logging.info(f"Generating 3D coordinates for {len(valid_matrices)} matrices...")
        coords3d = self._generate_coordinates(valid_matrices, valid_indices)

        features = self._extract_features(valid_matrices, valid_indices, valid_matrix_types, num_dims)
        if features is None:
            for idx in valid_indices:
                self.hyperdimensional_connections[idx] = []
            return self.hyperdimensional_connections

        projected_points, proj_dists, projection_norms = self._prepare_projections(valid_matrices, features)

        connections = self._run_connection_batches(
            features,
            projected_points,
            coords3d,
            valid_indices,
            context,
            batch_size_conn,
            use_memmap,
            memmap_dir,
            use_ann,
            ann_k,
            block_size,
            candidate_k,
            projection_norms,
            proj_dists,
        )

        self.hyperdimensional_connections = connections
        return self.hyperdimensional_connections