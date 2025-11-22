# Changelog

All notable changes to MatrixTransformer will be documented in this file.

## [0.1.0] - November 2025

### Fixed

#### Angular Tolerance Bug in `find_hyperdimensional_connections`

**Problem Overview:**

The `find_hyperdimensional_connections` method was returning very similar distances for data points projected to a unit hypersphere, causing multiple distance metrics to fail and produce zero or near-zero values.

**Affected Components:**

1. **Distance Metrics:**
   - `log_map_sphere` - Logarithm map on unit sphere
   - `local_distance_sphere` - Spherical distances between points

2. **Return Structure:**
   All fields in the connection targets dictionary were producing suboptimal results:
   ```python
   targets.append({
       "target_idx": valid_indices[tgt_idx],
       "high_dim_dist": float(hd_dist),
       "physical_dist": float(phys_dist),
       "ratio": float(ratio),
       "strength": float(similarity_val),
       "dimensions": significant_dimensions.tolist(),
       "log_map": v_ij.tolist() if hasattr(v_ij, 'tolist') else [],
       "log_map_norm": vnorm,
       "transported_log_map": v_ij_t.tolist() if hasattr(v_ij_t, 'tolist') else [],
       "reciprocal_angle": reciprocal_angle,
       "local_curvature": local_curvature,
       "local_energy": local_energy,
       "target_energy": target_energy,
       "energy_gradient": energy_gradient,
       "geodesic_error": geodesic_error,
       # VARIANCE FEATURES
       "source_projection_norm": src_projection_norm,
       "target_projection_norm": tgt_projection_norm,
       "norm_variance": norm_variance,
       "norm_variance_relative": norm_variance_relative
   })
   ```

**Root Cause:**

Points were already very close in angular terms before normalization (using 7.0 radius instead of 1.0). The tolerance threshold of `1e-7` was treating genuinely small but non-zero angles as if they were exactly zero, leading to degenerate calculations.

**Solution:**

Changed from **dot-product-based tolerance** to **angle-based tolerance**. This allows the system to properly distinguish between truly degenerate cases and valid small-angle scenarios, preserving the meaningful geometric relationships in high-dimensional space.

**Impact:**

- More accurate hyperdimensional connection detection
- Improved geodesic calculations on manifolds
- Better energy gradient and curvature estimates
- Reliable variance feature extraction

## [0.1.1] - November 22, 2025

### Added

### New: Streaming, memmap, ANN and element-level metadata in `find_hyperdimensional_connections`

- Streaming / batch processing:
   - New parameters: `batch_size_conn`, `block_size` enable chunked processing for large datasets to reduce memory usage and allow streaming workflows.
   - Trade-offs: smaller `batch_size_conn` reduces peak memory but increases IO/compute overhead.

- Memmap support:
   - `use_memmap=True` and `memmap_dir` allow memory-mapped intermediate storage for out-of-core operation.
   - Recommended: use an SSD-backed `memmap_dir` to reduce I/O latency.
   - Repository usage note: in the project we enable memmap for many small matrices using logic such as `use_memmap_flag = True if len(matrices) > 200 else False` to avoid unnecessary disk-backed IO for small runs.

- Approximate neighbor search (ANN):
   - `use_ann=True` with `ann_k` enables ANN-based candidate retrieval to speed high-dimensional neighbor search.
   - Trade-offs: speed vs exactness — tune `ann_k` (recommended 64–256) to balance recall and performance.

- Output filtering and richer outputs:
   - `candidate_k` and `top_k` control candidate limits and final returned connections.
   - Returned entries continue to include numeric connection fields (`high_dim_dist`, `physical_dist`, `ratio`, `strength`, `log_map`, energies, variance features); filtering is applied before final aggregation.

- Element-level semantic metadata attachment:
   - `_attach_semantic_metadata(summary_entry, src_idx, valid_indices[tgt_idx])` is called when `include_element_metadata=True` to enrich targets with element-level context (labels, original indices, small previews).
   - Metadata volume is limited via `preview_size` and `preview_cache_size`. Enable metadata only when needed; it increases payload and IO.

- Preview / cache options:
   - `preview_size` and `preview_cache_size` provide quick sampling and caching for large datasets.

- New parameters:
   - `batch_size_conn`, `use_memmap`, `memmap_dir`, `use_ann`, `ann_k`, `block_size`, `candidate_k`, `preview_size`, `preview_cache_size`, `include_element_metadata`.

- Backward compatibility:
   - Default behavior is unchanged if new flags are not used.
   - Note migration effects: enabling `use_memmap` or batch streaming changes IO; enabling `use_ann` may yield approximate neighbors.

- Example usage:
```python
summary = transformer.find_hyperdimensional_connections(
      num_dims=8,
      min_similarity=0.5,
      min_ratio=5.0,
      top_k=50,
      batch_size_conn=10000,
      use_memmap=True,
      memmap_dir='/tmp/matrix_mm',
      use_ann=True,
      ann_k=128,
      candidate_k=256,
      include_element_metadata=True,
      preview_size=512,
      preview_cache_size=128
)
```

- Performance recommendations:
   - For very large datasets (millions of points): `batch_size_conn` 5k–50k, `ann_k` 64–256, `candidate_k` 128–512, `preview_size` 256–1024.
   - For streaming/chunked runs with many small matrices the repository uses much smaller `batch_size_conn` values (e.g., `batch_size_conn=50`) and conditional memmap enabling (`use_memmap_flag = True if len(matrices) > 200 else False`). In those cases larger preview parameters have been used in practice (e.g., `preview_size=100000`, `preview_cache_size=20000`) when accompanied by `use_memmap` to avoid memory blow-up.
   - Smaller `batch_size_conn` reduces peak RAM but increases processing time and I/O; tune per environment.
   - For best throughput, combine `use_ann=True` with moderate `ann_k` and reasonable `batch_size_conn`.

- Testing & validation:
   - Unit/integration tests should cover memmap correctness, `_attach_semantic_metadata` index alignment, and acceptable recall for ANN.




