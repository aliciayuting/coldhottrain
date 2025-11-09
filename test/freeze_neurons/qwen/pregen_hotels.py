import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the top-k rows and columns by gradient L2 norm for each layer "
            "and export indices compatible with replace_linear_with_elementwise_preselected."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing layer gradient .npy files.",
    )
    parser.add_argument(
        "--model",
        default='qwen',
        help="model type",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=5.0,
        help="Fraction (percent) of rows/columns to retain based on L2 norm (default: 5%%).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to write the JSON output. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--pattern",
        default="*.npy",
        help="Glob pattern for gradient files (default: *.npy).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when searching for gradient files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error log messages.",
    )
    return parser.parse_args()


def setup_logging(quiet: bool) -> None:
    level = logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def top_k_indices(values: np.ndarray, percent: float) -> List[int]:
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if values.size == 0:
        return []
    if not (0.0 < percent <= 100.0):
        raise ValueError("percent must be in the range (0, 100]")

    k = int(np.ceil(values.size * (percent / 100.0)))
    k = max(1, min(values.size, k))
    partition_idx = np.argpartition(values, -k)[-k:]
    sorted_idx = partition_idx[np.argsort(values[partition_idx])[::-1]]
    return sorted_idx.tolist()


def build_weight_indices(
    num_rows: int, num_cols: int, rows: List[int], cols: List[int]
) -> List[List[int]]:
    selections = []
    if rows:
        row_array = np.asarray(rows, dtype=np.int64)
        all_cols = np.arange(num_cols, dtype=np.int64)
        rr, cc = np.meshgrid(row_array, all_cols, indexing="ij")
        selections.append(np.stack([rr.ravel(), cc.ravel()], axis=1))
    if cols:
        col_array = np.asarray(cols, dtype=np.int64)
        all_rows = np.arange(num_rows, dtype=np.int64)
        rr, cc = np.meshgrid(all_rows, col_array, indexing="ij")
        selections.append(np.stack([rr.ravel(), cc.ravel()], axis=1))
    if not selections:
        return []
    combined = np.vstack(selections)
    # Remove duplicates and sort by (row, col) for deterministic output.
    combined = np.unique(combined, axis=0)
    order = np.lexsort((combined[:, 1], combined[:, 0]))
    combined = combined[order]
    return combined.astype(np.int64).tolist()


def compute_layer_selection(matrix: np.ndarray, percent: float) -> Dict[str, object]:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D array for weight gradients.")

    row_norms = np.linalg.norm(matrix, axis=1)
    col_norms = np.linalg.norm(matrix, axis=0)

    top_rows = top_k_indices(row_norms, percent)
    top_cols = top_k_indices(col_norms, percent)

    weight_indices = build_weight_indices(matrix.shape[0], matrix.shape[1], top_rows, top_cols)
    bias_indices = sorted(set(int(idx) for idx in top_rows))

    return {
        "shape": list(map(int, matrix.shape)),
        "top_rows": [int(idx) for idx in top_rows],
        "top_cols": [int(idx) for idx in top_cols],
        "train_weight_indices": [[int(r), int(c)] for r, c in weight_indices],
        "train_bias_indices": bias_indices,
    }


def iter_gradient_files(root: Path, pattern: str, recursive: bool):
    if recursive:
        yield from (p for p in sorted(root.rglob(pattern)) if p.is_file())
    else:
        yield from (p for p in sorted(root.glob(pattern)) if p.is_file())


def main() -> None:
    args = parse_args()
    setup_logging(args.quiet)

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = list(iter_gradient_files(input_dir, args.pattern, args.recursive))
    if not files:
        logging.warning("No files matched pattern '%s' in %s", args.pattern, input_dir)

    selections: Dict[str, Dict[str, object]] = {}
    for path in files:
        rel_key = str(path.relative_to(input_dir))
        if 'embedding' in rel_key.lower():
            logging.info("Skipping embedding layer gradient %s", rel_key)
            continue
        try:
            matrix = np.load(path)
        except Exception as exc:
            logging.error("Failed to load %s: %s", rel_key, exc)
            continue

        if matrix.ndim != 2:
            logging.info("Skipping non-2D gradient %s with shape %s", rel_key, matrix.shape)
            continue

        selection = compute_layer_selection(matrix, args.percent)
        selections[rel_key] = selection
        logging.info(
            "Processed %s -> %d weights, %d biases",
            rel_key,
            len(selection["train_weight_indices"]),
            len(selection["train_bias_indices"]),
        )

    output_payload = {
        "percent": args.percent,
        "layers": selections,
    }

    serialized = json.dumps(output_payload, indent=2)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized)
        logging.info("Wrote selections to %s", output_path)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
