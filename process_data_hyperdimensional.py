"""
Process TSV files using load.py and send 4D features to find_hyperdimensional_connections.
Logs all outputs from the connection discovery process.
"""

## import logging
import json
from pathlib import Path
import numpy as np

from load import load_payload
from matrixtransformer import MatrixTransformer

## # Configure logging
## logging.basicConfig(
##     level=logging.DEBUG,
##     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
##     handlers=[
##         logging.FileHandler('hyperdimensional_connections.log'),
##         logging.StreamHandler()
##     ]
## )

## logger = logging.getLogger(__name__)


def _json_safe(value):
    """Recursively convert values to JSON-serialisable types."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def process_dataset_files(dataset_files=None, chunk_size=2048, overlap=512, encoding='utf-8'):
    """Process arbitrary dataset files and find hyperdimensional connections.

    Parameters
    - dataset_files: iterable of file paths (strings or Path). If None, auto-discovers
      common data files (`*.tsv, *.csv, *.txt, *.json`) in the current working directory.
    - chunk_size, overlap, encoding: forwarded to `load_payload`.
    """

    # Resolve file list
    if dataset_files is None:
        cwd = Path.cwd()
        patterns = ['*.tsv', '*.csv', '*.txt', '*.json']
        found = []
        for pat in patterns:
            found.extend(list(cwd.glob(pat)))
        dataset_files = sorted({str(p) for p in found})
    else:
        # allow caller to pass a single path string
        if isinstance(dataset_files, (str, Path)):
            dataset_files = [str(dataset_files)]
        else:
            dataset_files = [str(p) for p in dataset_files]
    
    # logger.info("=" * 80)
    # logger.info("Dataset File Processing and Hyperdimensional Connection Discovery")
    # logger.info("=" * 80)
    
    # Create shared registry for all datasets
    registry = None
    all_results = []
    all_connections = {}
    
    # Process each TSV file
    for i, tsv_path in enumerate(dataset_files):
        file_path = Path(tsv_path)
        
        if not file_path.exists():
            # logger.warning(f"File not found: {tsv_path}")
            continue
        
        # logger.info(f"\n{'='*80}")
        # logger.info(f"Processing file {i+1}/{len(dataset_files)}: {file_path.name}")
        # logger.info(f"{'='*80}")
        
        try:
            # Use load_payload to process the file
            # Generate a neutral dataset id that is not domain-specific
            safe_stem = file_path.stem.replace(' ', '_')
            dataset_id = f"dataset_{i}_{safe_stem}"

            # logger.info(f"Loading payload from: {file_path} (dataset_id={dataset_id})")
            result = load_payload(
                payload=file_path,
                dataset_id=dataset_id,
                registry=registry,
                chunk_size=chunk_size,
                overlap=overlap,
                encoding=encoding
            )
            
            # Use the registry from first result for subsequent loads
            if registry is None:
                registry = result.registry
            
            # Log LoadResult details
            # logger.info(f"\n--- LoadResult for {file_path.name} ---")
            # logger.info(f"Dataset ID: {dataset_id}")
            # logger.info(f"Source metadata: {json.dumps(result.source_metadata, indent=2)}")
            # logger.info(f"Features shape: {result.features.shape}")
            # logger.info(f"Features dtype: {result.features.dtype}")
            # logger.info(f"Number of chunks: {len(result.chunk_metadata)}")
            
            # Log manifest details
            # logger.info(f"\nManifest:")
            # logger.info(f"  Total chunks: {result.manifest.get('total_chunks', 'N/A')}")
            # logger.info(f"  Chunk size: {result.manifest.get('chunk_size', 'N/A')}")
            # logger.info(f"  Overlap: {result.manifest.get('overlap', 'N/A')}")
            # logger.info(f"  Base chunk: {result.manifest.get('base_chunk', 'N/A')}")
            
            # Log first few chunk metadata entries
            # logger.info(f"\nFirst 3 chunk metadata entries:")
            # for j, chunk_meta in enumerate(list(result.chunk_metadata)[:3]):
            #     # Convert bytes to hex string for JSON serialization
            #     serializable_meta = {}
            #     for key, value in chunk_meta.items():
            #         if isinstance(value, bytes):
            #             serializable_meta[key] = f"<bytes: {len(value)} bytes, hex: {value[:20].hex()}...>"
            #         else:
            #             serializable_meta[key] = value
            #     logger.info(f"  Chunk {j}: {json.dumps(serializable_meta, indent=4)}")
            
            chunk_metadata_list = list(getattr(result, 'chunk_metadata', []))

            # Persist element records to a separate JSON file to avoid huge inline blobs
            element_file_name = None
            try:
                element_file_name = f"element_records_{dataset_id}.json"
                element_file_path = Path(element_file_name)
                with open(element_file_path, 'w', encoding='utf-8') as ef:
                    json.dump(_json_safe(list(getattr(result, 'element_records', []))), ef, indent=2)
            except Exception as e:
                # logger.warning(f"Failed to write element records for {dataset_id}: {e}")
                pass

            all_results.append({
                'dataset_id': dataset_id,
                'file_name': file_path.name,
                'result': result,
                'features': result.features,
                'chunk_metadata': chunk_metadata_list,
                'element_records_file': element_file_name
            })

            matrices = []
            matrix_to_chunk_map = {}
            num_chunks = len(result.features)
            for chunk_idx in range(num_chunks):
                feature_vector = result.features[chunk_idx]
                chunk_meta = chunk_metadata_list[chunk_idx] if chunk_idx < len(chunk_metadata_list) else {}
                matrices.append(feature_vector)
                chunk_index = chunk_meta.get('chunk_index', chunk_meta.get('index', chunk_idx)) if isinstance(chunk_meta, dict) else chunk_idx
                matrix_to_chunk_map[chunk_idx] = chunk_index

            if not matrices:
                # logger.warning(f"No chunk features found for dataset {dataset_id}; skipping connection search")
                continue

            # logger.info(f"\n{'='*80}")
            # logger.info(f"Preparing hyperdimensional transformer for dataset {dataset_id}")
            # logger.info(f"Total matrices: {len(matrices)}")

            transformer = MatrixTransformer(dimensions=128)
            transformer.matrices = matrices

            # logger.info(f"\n{'='*80}")
            # logger.info(f"Finding Hyperdimensional Connections for dataset {dataset_id}")
            # logger.info(f"{'='*80}")

            # Allow memmap when processing large numbers of chunks to avoid memory blow-up
            use_memmap_flag = True if len(matrices) > 200 else False
            memmap_dir_to_use = None
            connections = transformer.find_hyperdimensional_connections(
                num_dims=8,
                min_similarity=0.3,
                min_ratio=2.0,
                top_k=None,
                batch_size_conn=50,
                use_memmap=use_memmap_flag,
                memmap_dir=memmap_dir_to_use,
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

            all_connections[dataset_id] = {
                'connections': connections,
                'matrix_to_chunk_map': matrix_to_chunk_map,
                'transformer': transformer,
                'element_records_file': element_file_name
            }

            # logger.info(f"\n{'='*80}")
            # logger.info(f"Hyperdimensional Connections Results for {dataset_id}")
            # logger.info(f"{'='*80}")
            # logger.info(f"Total matrices analysed: {len(connections)}")
            # logger.info(f"Matrices with connections: {sum(1 for c in connections.values() if c)}")

            # if hasattr(transformer, '_projection_norms'):
            #     logger.info(f"\n--- Variance Preservation Metrics ({dataset_id}) ---")
            #     logger.info(f"Projection norms stored: {len(transformer._projection_norms)}")
            #     logger.info(f"Mean projection norm: {transformer._projection_norm_mean:.6f}")
            #     logger.info(f"Std projection norm: {transformer._projection_norm_std:.6f}")
            #     logger.info(f"Min projection norm: {min(transformer._projection_norms):.6f}")
            #     logger.info(f"Max projection norm: {max(transformer._projection_norms):.6f}")

            # logger.info(f"\n{'='*80}")
            # logger.info(f"Detailed Connection Information ({dataset_id})")
            # logger.info(f"{'='*80}")
            # for src_idx, conns in sorted(connections.items())[:5]:
            #     if not conns:
            #         continue
            #     logger.info(f"\n--- Source Matrix {src_idx} ({len(conns)} connections) ---")
            #     for i, conn in enumerate(conns[:3], 1):
            #         logger.info(f"  Connection {i} to Matrix {conn['target_idx']}")
            #         logger.info(f"    Strength: {conn['strength']:.6f}")
            #         logger.info(f"    Physical distance: {conn['physical_dist']:.6f}")
            #         logger.info(f"    Hyperdimensional distance: {conn['hyperdimensional_dist']:.6f}")
            #         logger.info(f"    Ratio: {conn['ratio']:.6f}")
            #         if 'source_metadata' in conn:
            #             preview = conn['source_metadata'].get('content_preview', '')
            #             logger.info(f"    Source preview: {preview[:200]}")
            #         if 'target_metadata' in conn:
            #             preview = conn['target_metadata'].get('content_preview', '')
            #             logger.info(f"    Target preview: {preview[:200]}")

            # logger.info(f"Successfully processed {file_path.name}")
            
        except Exception as e:
            # logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
            continue
    
    if not all_connections:
        # logger.error("No connections generated from any dataset!")
        return None

    # logger.info(f"\n{'='*80}")
    # logger.info(f"Aggregating connection results across {len(all_connections)} datasets")
    # logger.info(f"{'='*80}")

    output_file = "hyperdimensional_connections_output.json"
    output_payload = {
        'metadata': {
            'total_files_processed': len(all_results),
            'files_processed': [entry['file_name'] for entry in all_results],
            'element_records_files': []  # Track all element record files
        },
        'datasets': {}
    }

    for entry in all_results:
        dataset_id = entry['dataset_id']
        if dataset_id not in all_connections:
            continue

        conn_bundle = all_connections[dataset_id]
        connections = conn_bundle['connections']
        transformer = conn_bundle['transformer']
        chunk_metadata_list = entry.get('chunk_metadata', [])

        # Save element records to separate file
        element_records_file = f"element_records_{dataset_id}.json"
        element_records_payload = {
            'dataset_id': dataset_id,
            'file_name': entry['file_name'],
            'connections_file': output_file,  # Link back to main file
            'total_chunks': len(chunk_metadata_list),
            'element_records': []
        }

        # Extract element records from registry if available
        if registry is not None:
            try:
                import load as load_module
                elements = list(load_module.get_element_records(registry, dataset_id))
                element_records_payload['element_records'] = [_json_safe(elem) for elem in elements]
                element_records_payload['total_elements'] = len(elements)
            except Exception as exc:
                logger.warning(f"Could not extract element records for {dataset_id}: {exc}")
                element_records_payload['total_elements'] = 0

        # Write element records to separate file
        try:
            with open(element_records_file, 'w', encoding='utf-8') as elem_f:
                json.dump(element_records_payload, elem_f, indent=2)
            # logger.info(f"Element records for {dataset_id} saved to {element_records_file}")
            output_payload['metadata']['element_records_files'].append(element_records_file)
        except Exception as exc:
            # logger.error(f"Failed to write element records file {element_records_file}: {exc}")
            pass

        dataset_summary = {
            'file_name': entry['file_name'],
            'element_records_file': element_records_file,  # Cross-reference
            'total_matrices': len(connections),
            'matrices_with_connections': sum(1 for c in connections.values() if c),
            'connections': {}
        }

        variance_metrics = {}
        if hasattr(transformer, '_projection_norm_mean'):
            variance_metrics['projection_norm_mean'] = float(transformer._projection_norm_mean)
        if hasattr(transformer, '_projection_norm_std'):
            variance_metrics['projection_norm_std'] = float(transformer._projection_norm_std)
        if hasattr(transformer, '_projection_norms'):
            variance_metrics['projection_norms_sample'] = [float(x) for x in transformer._projection_norms[:50]]
        if variance_metrics:
            dataset_summary['variance_metrics'] = variance_metrics

        for src_idx, conn_list in sorted(connections.items()):
            dataset_summary['connections'][str(src_idx)] = [_json_safe(conn) for conn in conn_list]

        output_payload['datasets'][dataset_id] = dataset_summary

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, indent=2)

    # logger.info(f"Connections saved to {output_file}")
    # logger.info(f"Element records saved to {len(output_payload['metadata']['element_records_files'])} separate files")
    # logger.info(f"\n{'='*80}")
    # logger.info(f"Processing Complete!")
    # logger.info(f"{'='*80}")

    return {
        'registry': registry,
        'results': all_results,
        'connections': all_connections
    }


if __name__ == "__main__":
    import sys
    # logger.info("\nStarting dataset processing and hyperdimensional connection discovery...\n")

    # Accept dataset file paths from CLI arguments (skip script name)
    cli_args = sys.argv[1:]
    if cli_args:
        # logger.info(f"Received {len(cli_args)} dataset file path(s) from command line.")
        result = process_dataset_files(dataset_files=cli_args)
    else:
        # logger.info("No dataset file paths provided via CLI; using auto-discovery.")
        result = process_dataset_files()

    # if result:
    #     logger.info("\n✓ All processing completed successfully!")
    #     logger.info("✓ Check 'hyperdimensional_connections.log' for detailed logs")
    #     logger.info("✓ Check 'hyperdimensional_connections_output.json' for connection data")
    # else:
    #     logger.error("\n✗ Processing failed!")
