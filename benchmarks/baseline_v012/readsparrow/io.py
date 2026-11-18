"""Strict, lossless input parsing and small reproducibility helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        try:
            import openpyxl
        except ImportError as exc:
            raise ValueError("XLSX input needs readsparrow[excel]; TSV/CSV needs no Excel dependency") from exc
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            rows = iter(book.active.values)
            header = [str(x).strip() if x is not None else "" for x in next(rows)]
            values = [dict(zip(header, ["" if x is None else str(x) for x in row])) for row in rows if any(x is not None for x in row)]
        finally:
            book.close()
    else:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter="," if path.suffix.lower() == ".csv" else "\t")
            header = reader.fieldnames or []
            values = list(reader)
    if not header or len(header) != len(set(header)) or any(not x for x in header):
        raise ValueError("Missing, empty or duplicate column names")
    if not values or any(None in row or any(v is None for v in row.values()) for row in values):
        raise ValueError("Empty table or ragged rows")
    return header, values


def write_tsv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def new_output_dir(path):
    path = Path(path)
    # An empty caller-created directory is allowed; existing results are never overwritten.
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be new or empty, refusing to overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def integer_count(text):
    try:
        value = Decimal(str(text))
        if not value.is_finite() or value < 0 or value != value.to_integral_value() or value > np.iinfo(np.int64).max:
            raise ValueError
        return int(value)
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Counts must be finite, non-negative int64 integers, received {text!r}") from None


@dataclass
class CountTable:
    ids: list[str]
    genes: list[str]
    samples: list[str]
    counts: np.ndarray

    def __post_init__(self):
        if not self.ids or not self.samples or len(self.ids) != len(set(self.ids)) or any(not x for x in self.ids):
            raise ValueError("Nonempty, unique guide IDs and nonempty samples required")
        if len(self.samples) != len(set(self.samples)) or any(not x for x in self.samples):
            raise ValueError("Sample names must be nonempty and unique")
        if len(self.genes) != len(self.ids) or any(not x for x in self.genes):
            raise ValueError("Every guide needs an explicit gene/control annotation")
        a = np.asarray(self.counts)
        if a.shape != (len(self.ids), len(self.samples)) or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("Count matrix must be non-negative integers with matching dimensions")
        if np.any(a.astype(object).sum(axis=0) > np.iinfo(np.int64).max):
            raise ValueError("Sample count total overflows int64")
        self.counts = a.astype(np.int64)


def load_counts(path, samples=None):
    header, rows = read_rows(path)
    id_col = next((x for x in ("sgRNA", "ID", "guide_id") if x in header), None)
    gene_col = next((x for x in ("Gene", "gene_symbol", "gene") if x in header), None)
    if id_col is None or gene_col is None:
        raise ValueError("Expected sgRNA/ID/guide_id and Gene/gene_symbol/gene columns")
    samples = list(samples) if samples is not None else [x for x in header if x not in {id_col, gene_col, "seq", "Sequence", "sequence", "contains_GTTTT"}]
    if not samples or any(s not in header for s in samples):
        raise ValueError("Missing selected count columns")
    return CountTable([r[id_col] for r in rows], [r[gene_col] for r in rows], samples,
                      np.array([[integer_count(r[s]) for s in samples] for r in rows], dtype=np.int64))


def save_counts(path, table):
    write_tsv(path, ["sgRNA", "Gene", *table.samples],
              ({"sgRNA": gid, "Gene": gene, **dict(zip(table.samples, map(int, row)))}
               for gid, gene, row in zip(table.ids, table.genes, table.counts)))
