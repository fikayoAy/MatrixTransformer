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



