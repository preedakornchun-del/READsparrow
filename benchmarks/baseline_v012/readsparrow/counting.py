"""Length-parameterized legacy-compatible exact anchor-bounded extraction.

Compatibility mode intentionally retains first-left/first-right/first-match rules.
It does not claim ambiguity rejection or mutation rescue; these need new validation.
"""
from __future__ import annotations

import gzip
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import CountTable, read_rows


def reverse_complement(seq):
    return seq.translate(bytes.maketrans(b"ACGT", b"TGCA"))[::-1]


@dataclass
class Library:
    ids: list[str]
    genes: list[str]
    sequences: list[bytes]

    def __post_init__(self):
        if not self.ids or len(self.ids) != len(self.genes) or len(self.ids) != len(self.sequences):
            raise ValueError("Library is empty or inconsistent")
        if len(set(self.ids)) != len(self.ids) or any(not x for x in self.ids):
            raise ValueError("Library guide IDs must be unique and nonempty")
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("Library sequences must be unique")
        lengths = {len(x) for x in self.sequences}
        if len(lengths) != 1 or next(iter(lengths)) < 1 or any(set(x) - set(b"ACGT") for x in self.sequences):
            raise ValueError("Use a uniform guide length with A/C/G/T sequences")
        if any(not x for x in self.genes):
            raise ValueError("Provide gene annotations, or NA for controls")

    @property
    def length(self):
        return len(self.sequences[0])


def load_library(path):
    header, rows = read_rows(path)
    aliases = [("sgRNA", "ID", "guide_id"), ("Gene", "gene_symbol", "gene"), ("Sequence", "seq", "sequence")]
    cols = [next((x for x in options if x in header), None) for options in aliases]
    if any(x is None for x in cols):
        raise ValueError("Library requires guide ID, gene annotation and sequence columns")
    try:
        return Library([r[cols[0]] for r in rows], [r[cols[1]] for r in rows], [r[cols[2]].upper().encode("ascii") for r in rows])
    except UnicodeEncodeError as exc:
        raise ValueError("Non-ASCII guide sequence") from exc


def fastq_records(path, max_reads=None):
    if max_reads is not None and max_reads < 1:
        raise ValueError("max_reads must be positive")
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as handle:
        n = 0
        while max_reads is None or n < max_reads:
            h = handle.readline()
            if not h:
                return
            sequence, plus, quality = handle.readline(), handle.readline(), handle.readline()
            n += 1
            if not sequence or not plus or not quality or not h.startswith(b"@") or not plus.startswith(b"+"):
                raise ValueError(f"Malformed/truncated four-line FASTQ at record {n} in {path}")
            seq, qual = sequence.rstrip(b"\r\n").upper(), quality.rstrip(b"\r\n")
            if not seq or len(seq) != len(qual) or set(seq) - set(b"ACGTN"):
                raise ValueError(f"Invalid bases or sequence/quality lengths at record {n}")
            yield seq


class ExactExtractor:
    def __init__(self, library, orientation="forward", left=b"CACCG", right=b"GTTTT", policy="legacy", max_extra=32):
        if orientation not in ("forward", "reverse"):
            raise ValueError("Orientation must be explicitly forward or reverse")
        if not left or not right or set(left + right) - set(b"ACGT"):
            raise ValueError("Anchors must be nonempty A/C/G/T bytes")
        self.length = library.length
        if policy not in {"legacy", "unique"} or max_extra < 0:
            raise ValueError("policy must be legacy/unique and max_extra non-negative")
        self.policy, self.max_extra = policy, max_extra
        self.left, self.right = (left, right) if orientation == "forward" else (reverse_complement(right), reverse_complement(left))
        sequences = library.sequences if orientation == "forward" else [reverse_complement(s) for s in library.sequences]
        self.lookup = {s: i for i, s in enumerate(sequences)}

    def match(self, sequence):
        if self.policy == "unique":
            return self._unique_match(sequence)
        left = sequence.find(self.left)
        if left < 0:
            return None, "no_left_anchor"
        start = left + len(self.left)
        right = sequence.find(self.right, start + self.length)
        if right < 0:
            return None, "no_usable_right_anchor"
        for offset in range(start, right - self.length + 1):
            index = self.lookup.get(sequence[offset:offset + self.length])
            if index is not None:
                return index, "assigned_exact"
        return None, "anchor_region_no_exact_reference_match"

    def _unique_match(self, sequence):
        """Enumerate all exact left anchors; refuse competing guide identities.

        Use the nearest usable right anchor after L. A bounded extra span avoids
        searching arbitrary distant guide-like sequences. This is not alignment,
        UMI handling, guide-mismatch correction or anchor-mutation rescue.
        """
        matches, saw_left, saw_right, saw_long = set(), False, False, False
        position = sequence.find(self.left)
        while position >= 0:
            saw_left = True
            start = position + len(self.left)
            right = sequence.find(self.right, start + self.length)
            if right >= 0:
                saw_right = True
                if right - start <= self.length + self.max_extra:
                    for offset in range(start, right - self.length + 1):
                        index = self.lookup.get(sequence[offset:offset + self.length])
                        if index is not None:
                            matches.add(index)
                else:
                    saw_long = True
            position = sequence.find(self.left, position + 1)
        if len(matches) > 1:
            return None, "ambiguous_multiple_guides"
        if matches:
            return next(iter(matches)), "assigned_exact"
        return None, ("no_left_anchor" if not saw_left else "no_usable_right_anchor" if not saw_right else
                      "anchor_span_exceeds_limit" if saw_long else "anchor_region_no_exact_reference_match")


def count_fastq(library, path, sample="sample", orientation="forward", left=b"CACCG", right=b"GTTTT", max_reads=None,
                policy="legacy", max_extra=32):
    extractor = ExactExtractor(library, orientation, left, right, policy, max_extra)
    counts = np.zeros(len(library.ids), dtype=np.int64)
    categories = Counter()
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    for seq in fastq_records(path, max_reads):
        index, reason = extractor.match(seq)
        categories[reason] += 1
        if index is not None:
            counts[index] += 1
    total, assigned = sum(categories.values()), int(counts.sum())
    if total == 0:
        raise ValueError("FASTQ contains no records")
    metrics = {"total_reads": total, "mapped_reads": assigned, "mapping_rate_percent": 100 * assigned / total,
               "categories": dict(categories), "orientation": orientation, "guide_length": library.length,
               "left_anchor": extractor.left.decode(), "right_anchor": extractor.right.decode(),
               "algorithm": "legacy-compatible-first-exact-window" if policy == "legacy" else "unique-exact-anchor-bounded",
               "policy": policy, "max_extra": max_extra if policy == "unique" else None, "max_reads": max_reads,
               "loop_wall_seconds": time.perf_counter() - start_wall, "loop_cpu_seconds": time.process_time() - start_cpu,
               "quality_scores_used": False, "ambiguous_matches_rejected": policy == "unique", "mutation_rescue": False}
    return CountTable(library.ids, library.genes, [sample], counts[:, None]), metrics
