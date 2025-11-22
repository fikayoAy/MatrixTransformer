
import csv
import json
from pathlib import Path
import sys

def load_data_rows(data_path):
    """
    Load rows from any text-based file, skipping header lines if present.
    Returns a list of strings (lines), with no format assumptions.
    """
    data_path = Path(data_path)
    rows = []
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
        # Heuristic: skip first 2 lines if file has more than 2 lines (possible headers)
        if len(lines) > 2:
            data_lines = lines[2:]
        else:
            data_lines = lines
        for line in data_lines:
            rows.append(line.rstrip("\r\n"))
    return rows

def build_element_index_map(data_rows, element_records_file):
    """
    Map element_index to the corresponding row (string) in the original data file.
    """
    with open(element_records_file, encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "element_records" in data:
            records = data["element_records"]
        else:
            records = data
    index_map = {}
    for rec in records:
        idx = rec.get("element_index")
        if idx is not None and 0 <= idx < len(data_rows):
            index_map[idx] = data_rows[idx]
    return index_map

def main():

    base_dir = Path(__file__).parent
    output_json = base_dir / "hyperdimensional_connections_output.json"

    with open(output_json, encoding="utf-8") as f:
        output = json.load(f)

    # Build dataset_id -> (original data file, element_records file)
    dataset_info = {}
    for dsid, dsdata in output.get("datasets", {}).items():
        # Try to get the original data file from the summary
        file_name = dsdata.get("file_name")
        element_records_file = dsdata.get("element_records_file")
        if not file_name or not element_records_file:
            continue
        dataset_info[dsid] = {
            "data_file": base_dir / file_name,
            "element_records_file": base_dir / element_records_file
        }

    # Load all data rows and build element index maps
    element_maps = {}
    for dsid, info in dataset_info.items():
        data_file = info["data_file"]
        element_records_file = info["element_records_file"]
        if not data_file.exists() or not element_records_file.exists():
            continue
        data_rows = load_data_rows(data_file)
        element_maps[dsid] = build_element_index_map(data_rows, element_records_file)


    # Discover all metric keys present in any connection so we can expose them as columns
    metric_keys = set()
    for dsdata in output.get("datasets", {}).values():
        for conn_list in dsdata.get("connections", {}).values():
            for conn in conn_list:
                for k in conn.keys():
                    if k in ("target_idx", "target_dataset_id", "source_metadata", "target_metadata"):
                        continue
                    metric_keys.add(k)
    metric_keys = sorted(metric_keys)

    # Export connections as JSON list of rows instead of CSV
    json_path = base_dir / "connections_linked.json"
    rows = []
    for dsid, dsdata in output.get("datasets", {}).items():
        connections = dsdata.get("connections", {})
        for src_idx_str, conn_list in connections.items():
            try:
                src_idx = int(src_idx_str)
            except Exception:
                continue
            src_row = element_maps.get(dsid, {}).get(src_idx, "")
            src_row_str = src_row
            for conn in conn_list:
                tgt_idx = conn.get("target_idx")
                tgt_dsid = conn.get("target_dataset_id", dsid)
                tgt_row = element_maps.get(tgt_dsid, {}).get(tgt_idx, "")
                tgt_row_str = tgt_row
                # Build a dict with all discovered metric keys
                entry = {
                    "source_dataset_id": dsid,
                    "source_element_index": src_idx,
                    "source_row": src_row_str,
                }
                for k in metric_keys:
                    entry[k] = conn.get(k, "")
                # add previews and target info
                src_preview = ""
                tgt_preview = ""
                if isinstance(conn.get("source_metadata"), dict):
                    src_preview = conn.get("source_metadata", {}).get("content_preview", "")
                if isinstance(conn.get("target_metadata"), dict):
                    tgt_preview = conn.get("target_metadata", {}).get("content_preview", "")
                entry.update({
                    "source_preview": src_preview,
                    "target_preview": tgt_preview,
                    "target_dataset_id": tgt_dsid,
                    "target_element_index": tgt_idx,
                    "target_row": tgt_row_str,
                })
                rows.append(entry)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(rows, jf, ensure_ascii=False, indent=2)

    print(f"JSON export complete: {json_path}")

if __name__ == "__main__":
    main()