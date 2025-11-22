# find_hyperdimensional_connections — Purpose, initialization, and outputs

This document explains what `find_hyperdimensional_connections` computes, how to initialize and call it following the pattern in `process_data_hyperdimensional.py`, what each metric in a returned connection entry means, what `_attach_semantic_metadata(...)` typically adds, and how `export_connections_to_json.py` consumes the outputs produced by `process_data_hyperdimensional.py`.

## 1) High-level purpose

- `find_hyperdimensional_connections` compares high-dimensional feature vectors (matrix/chunk embeddings) to detect pairs of items that are similar in embedding space but also related in original (physical) space. It returns per-source lists of connection summaries (dictionaries) describing candidate target matrices and a set of numeric diagnostics.

## 2) Initialization & example call (matching `process_data_hyperdimensional.py`)

The pipeline in `process_data_hyperdimensional.py` follows these steps before calling `find_hyperdimensional_connections`:

- Load a data file with `load_payload(...)` which returns a `result` object containing `result.features` (an iterable/array of feature vectors), `result.chunk_metadata` and optionally `result.element_records`.
- Build a list `matrices` from `result.features` and a `matrix_to_chunk_map` mapping local matrix index → original chunk index.
- Create a `MatrixTransformer` and assign `transformer.matrices = matrices`.
- Call `transformer.find_hyperdimensional_connections(...)` using parameters similar to the script's call.

Minimal pseudo-code (adapted from `process_data_hyperdimensional.py`):

```py
from matrixtransformer import MatrixTransformer

# matrices: list/iterable of feature vectors (from load_payload result)
transformer = MatrixTransformer(dimensions=128)
transformer.matrices = matrices

connections = transformer.find_hyperdimensional_connections(
    num_dims=8,
    min_similarity=0.3,
    min_ratio=2.0,
    top_k=None,
    batch_size_conn=50,
    use_memmap=(len(matrices) > 200),
    memmap_dir=None,
    use_ann=False,
    ann_k=128,
    block_size=1024,
    candidate_k=256,
    registry=registry,
    dataset_id=dataset_id,
    matrix_to_chunk_map=matrix_to_chunk_map,
    include_element_metadata=True,
    preview_size=100000,
    preview_cache_size=20000
)

# `connections` is typically a dict mapping source_idx -> list[summary_entry]
```

Parameters above are taken directly from `process_data_hyperdimensional.py`. Tune `min_similarity`, `min_ratio`, `candidate_k`, and `top_k` to change sensitivity and quantity of results.

## 3) Output shape and storage (how `process_data_hyperdimensional.py` saves results)

- `connections` (returned by `find_hyperdimensional_connections`) is stored per-dataset in `all_connections[dataset_id]['connections']`.
- The script writes a top-level JSON file `hyperdimensional_connections_output.json` with structure roughly:

```json
{
  "metadata": { ... },
  "datasets": {
    "dataset_0_somefile": {
      "file_name": "somefile.tsv",
      "element_records_file": "element_records_dataset_0_somefile.json",
      "total_matrices": N,
      "matrices_with_connections": M,
      "connections": { "0": [ {...}, {...} ], "1": [...], ... }
    },
    ...
  }
}
```

- `process_data_hyperdimensional.py` also writes per-dataset `element_records_{dataset_id}.json` (if present) so downstream exporters can map element indices back to original rows.

## 4) Metric keys returned by `find_hyperdimensional_connections` — meanings

Below are common keys present in each connection `summary_entry` and short definitions.

- `source_idx` — integer index of the source matrix (local index in `transformer.matrices`).
- `target_idx` — integer index of the target matrix (local index, or mapped via `matrix_to_chunk_map`).
- `high_dim_dist` / `hyperdimensional_dist` — distance between embeddings in high-dimensional (feature) space. Lower means more similar.
- `physical_dist` — distance in original (physical) space (e.g., Euclidean distance computed from original coordinates or domain-specific metric). Lower means physically closer.
- `ratio` — `physical_dist / hyperdim_dist`. Values >> 1 indicate the physical distance is large relative to embedding distance (possible interesting relation); values near 1 indicate parity between spaces.
- `strength` — similarity score used to rank connections (higher = stronger link). Often derived from a normalized kernel or cosine/similarity metric in feature space.
- `dimensions` — list of dimension indices used for this comparison (context-specific; may be the projection axes used to compute this connection).
- `log_map_norm` — norm of the log map/vector used to transport from source to target in the embedding manifold (often used to compare manifold distance to physical distance).
- `reciprocal_angle` — geometric angle between transport vectors in forward/backward directions (0.0 if not computed). Lower indicates more reciprocal alignment.
- `local_curvature` — local curvature proxy computed as `(hd_dist - phys_dist) / (phys_dist + eps)`; measures discrepancy between manifold and physical geometry locally.
- `local_energy` — energy associated with the source/local patch (domain-specific scalar). Larger values may indicate higher activity/importance.
- `target_energy` — same energy metric for the target node.
- `energy_gradient` — `target_energy - local_energy`: positive if energy increases towards the target.
- `geodesic_error` — `abs(log_map_norm - phys_dist)`: mismatch between transported manifold norm and physical distance.
- `source_projection_norm` / `target_projection_norm` — norms of projections used when computing alignments or directional searches.
- `norm_variance` — variance of projection norms across dimensions (absolute variance).
- `norm_variance_relative` — variance normalized relative to a baseline (e.g., mean), used to compare anisotropy in projections.

Notes:
- Many of the names above are implementation-specific (naming in the codebase may vary). When interpreting results, check whether `physical_dist` is measured in original units (pixels, microns, etc.).
- `strength` and `high_dim_dist` are the primary ranking signals; `ratio`, `local_curvature`, and `energy_gradient` are heuristics for prioritising surprising or semantically meaningful links.

## 5) What `_attach_semantic_metadata(summary_entry, src_idx, tgt_idx)` does

- `_attach_semantic_metadata` is a helper that enriches the numeric `summary_entry` with semantic/contextual metadata used for display, export, or downstream processing. Typical additions include:
  - `source_metadata` and `target_metadata`: dictionaries containing element-level metadata (labels, small content previews, entity ids, textual snippets, dataset ids, etc.).
  - `original`: sometimes the original low-level edge payload or index mapping used to create the summary.
  - human-friendly `labels` or `entity` fields when the registry or `element_records` is available.

- In `process_data_hyperdimensional.py` the code calls `_attach_semantic_metadata(summary_entry, src_idx, valid_indices[tgt_idx])` before storing `summary_entry` in the connections structure. This means the `hyperdimensional_connections_output.json` file contains both numeric diagnostics and the semantic metadata required by `export_connections_to_json.py`.

Important: the precise fields inserted by `_attach_semantic_metadata` depend on your project's `load`/`registry` implementation. The exporter expects either `source_metadata`/`target_metadata` or enough information to map indices back to a text row via the separate `element_records_*.json` files.

## 6) How `process_data_hyperdimensional.py` structures files for export

- For each processed dataset `dataset_id`, the script saves:
  - `element_records_{dataset_id}.json` — element-level metadata extracted from `load_payload`.
  - The main `hyperdimensional_connections_output.json` (aggregates all datasets).

- `hyperdimensional_connections_output.json` stores a per-dataset `connections` mapping where keys are stringified `source_idx` and values are arrays of `summary_entry` dicts (each entry contains metrics and semantic metadata).

## 7) How `export_connections_to_json.py` consumes the outputs

`export_connections_to_json.py` performs the following flow:

1. Load `hyperdimensional_connections_output.json` (the script uses `base_dir = Path(__file__).parent`).
2. For each dataset entry in `output['datasets']`, the script reads the dataset `file_name` and `element_records_file` fields to build a map from `element_index` → original input row string (via `load_data_rows()` and `build_element_index_map()`).
3. It scans all connection entries to collect metric keys (so the export contains a consistent set of columns across all rows).
4. It writes `connections_linked.json` as an array of flattened entries. Each flattened row includes:
   - `source_dataset_id`, `source_element_index`, `source_row`
   - all discovered metric keys (`strength`, `physical_dist`, `hyperdimensional_dist`, `ratio`, `local_energy`, etc.)
   - `source_preview`, `target_preview`, `target_dataset_id`, `target_element_index`, `target_row`.

Notes about expectations and ordering:
- `export_connections_to_json.py` resolves `data_file` and `element_records_file` using `base_dir / file_name` and `base_dir / element_records_file`. That means the exporter expects the original data files and the `element_records_*.json` files to be in the same folder (or reachable via the stored relative paths). If you move outputs to another directory, update paths in the `hyperdimensional_connections_output.json` or run the exporter from the correct base directory.
- The exporter iterates datasets in the order of keys in the JSON; while JSON object key ordering is preserved in modern Python, do not rely on ordering for semantic correctness. However, the mapping of `dataset_id` → `file_name` is essential: if dataset outputs and element records are placed in different folders or renamed, the exporter will not find corresponding files unless paths are adjusted. In short: keep the outputs produced by `process_data_hyperdimensional.py` together with the original data (or update paths) — otherwise the exported `target_row`/`source_row` fields will be empty.

## 8) Practical tips

- If you plan to share `hyperdimensional_connections_output.json` and the element records with collaborators, zip the `hyperdimensional_connections_output.json`, `element_records_*.json` files, and the original data files (`*.tsv`) together so `export_connections_to_json.py` can resolve file paths reliably.
- If you need a deterministic ordering of exported rows, post-process `connections_linked.json` (it is written as an array) and sort by `source_dataset_id` / `source_element_index` / `strength` as needed.
- For large datasets, prefer `use_memmap=True` and `top_k` limiting to reduce RAM and output size.

## 9) Where to look in the codebase

- `process_data_hyperdimensional.py` — shows how `load_payload` → `MatrixTransformer` → `find_hyperdimensional_connections` are wired and how outputs are written to `hyperdimensional_connections_output.json` and `element_records_*.json`.
- `export_connections_to_json.py` — shows how the exporter expects files to be arranged and how it flattens connection dicts into `connections_linked.json`.
- `matrixtransformer.py` — contains implementation of `find_hyperdimensional_connections` and helper functions (projection, energy, curvature calculations). Look for the function definition to confirm exact field names if you need precise typing.

If you'd like, I can:

- Patch `discover_tree_viz.html` to use the same `element_records` format for labels, or
- Generate a small sample workflow README or script that demonstrates producing and exporting connections for a single dataset (including exact commands to run). 

---
End of document.

## 10) CLI usage (how to run the pipeline)

- Process one or more data files with `process_data_hyperdimensional.py`. The script accepts file paths on the command line; if no paths are provided it will auto-discover common data files in the current directory. Example (textual datasets):

```powershell
# from the repo root
python process_data_hyperdimensional.py subset_object_Cancer_epithelial_subcluster.tsv subset_object_Myeloid_subcluster.tsv subset_object_T_cells_subcluster.tsv
```

- Notes:
  - The processor accepts arbitrary file types but the current tooling is geared to textual documents (TSV/CSV/JSON) via `load_payload`.
  - You do not need to edit `process_data_hyperdimensional.py` to run it — it will write `hyperdimensional_connections_output.json` and per-dataset `element_records_*.json` files into the working directory.

- Export discovered connections to a flattened JSON with `export_connections_to_json.py` (it auto-detects the output file and per-dataset element records saved by the processor):

```powershell
python export_connections_to_json.py
```

  - This creates `connections_linked.json` in the same directory.
  - Make sure you run the exporter from the same directory where `hyperdimensional_connections_output.json` and `element_records_*.json` were written (or move the files together) so the exporter can resolve original rows.

  ### Metric distributions & validation artifacts

  - The repository includes precomputed metric distributions and a validation report for the example datasets under the `data&metrics-val` folder:
    - `data&metrics-val/validation_report.txt` — validation summary produced by `validate_hyperdimensional_metrics.py`.
    - `data&metrics-val/diagnostics/metrics_distributions.png` — visual diagnostics for key metrics (physical_dist, log_map_norm, energy_gradient, norm variance, ratio, strength).
  
  #### Metric distributions (visual)

  ![Metric distributions](../data&metrics-val/diagnostics/metrics_distributions.png)

  #### Data source & citation

  The example datasets used for the demo and validation were derived from the processed data related to:

  Wu, S.Z., Al-Eryani, G., Roden, D.L. et al. "A single-cell and spatially resolved atlas of human breast cancers." Nat Genet 53, 1334–1347 (2021). https://doi.org/10.1038/s41588-021-00911-1

  If you use data from this repository, please consider citing that study.

  Study summary (from the public dataset):

  Breast cancers are complex cellular ecosystems where heterotypic interactions play central roles in disease progression and response to therapy. This work presents a single-cell and spatially resolved transcriptomics analysis of human breast cancers, introduces SCSubtype for intrinsic subtype classification, and provides high-resolution immune profiling (CITE-seq). The authors identify spatial stromal-immune niches, new macrophage populations, and present single-cell signatures used to stratify cohorts into ecotypes with distinct compositions and clinical outcomes.

  Dataset portal: https://singlecell.broadinstitute.org/single_cell/study/SCP1039/a-single-cell-and-spatially-resolved-atlas-of-human-breast-cancers#study-summary

  - To run the demo using those example datasets (they are included in `data&metrics-val`), run:

  ```powershell
  python process_data_hyperdimensional.py data&metrics-val/subset_object_Cancer_epithelial_subcluster.tsv \
      data&metrics-val/subset_object_Myeloid_subcluster.tsv data&metrics-val/subset_object_T_cells_subcluster.tsv
  ```

    - After processing, run the validator to reproduce the validation report and diagnostics:

  ```powershell
  python validate_hyperdimensional_metrics.py
  ```

    - The outputs will be written to the same working directory (or the data folder if you run the scripts from there) and will match the example artifacts in `data&metrics-val`.

## 11) Validation script: `validate_hyperdimensional_metrics.py`

- Purpose: performs a suite of geometric and statistical checks on the connections output to surface calculation issues and produce simple diagnostics/plots.
- How to run (from the directory containing `hyperdimensional_connections_output.json`):

```powershell
python validate_hyperdimensional_metrics.py
```

- Outputs:
  - `validation_report.txt` — human-readable summary of checks and any critical issues found.
  - `diagnostics/metrics_distributions.png` — summary plots for key metrics (physical_dist, log_map_norm, energy_gradient, norm variance, ratio, strength).

- Key validations performed:
  - Log map vs physical distance consistency (checks `log_map_norm` ≈ `physical_dist`).
  - Projection norm statistics (compares `source_projection_norm`/`target_projection_norm` to expected radius).
  - Local curvature sanity checks (`local_curvature`).
  - Energy consistency (`local_energy`, `target_energy`, `energy_gradient`).
  - Geodesic error (`geodesic_error`).
  - Norm variance checks (`norm_variance`, `norm_variance_relative`).
  - Reciprocal angle range checks (`reciprocal_angle`).
  - Ratio consistency (`ratio = physical_dist / high_dim_dist`).

- Notes:
  - Run this after `process_data_hyperdimensional.py` (and optionally after `export_connections_to_json.py`) to validate produced metrics.
  - The validator expects the aggregated `hyperdimensional_connections_output.json` layout produced by the processor.


