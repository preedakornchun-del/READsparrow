import gzip

import pandas as pd
import pytest

import readsparrow as rs
from readsparrow.cli import main


ADAPTER = "TCTTGTGGAAAGGACGAAACACCN"
ACTUAL_ADAPTER = ADAPTER[:-1] + "G"
GUIDE = "ACGTGCACTGATCGATGACA"
UNIQUE = "T" * 20


def write_fastq(path, reads):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as handle:
        for index, sequence in enumerate(reads):
            handle.write(f"@r{index}\n{sequence}\n+\n{'I' * len(sequence)}\n")
    return path


def reference_frame():
    return pd.DataFrame({
        "sgRNA": ["g1", "g1_alias", "unique"],
        "Gene": ["GENE1", "GENE2", "GENE3"],
        "Sequence": [GUIDE, GUIDE, UNIQUE],
    })


def test_count_flex_python_api_preserves_duplicate_sequence_semantics(tmp_path):
    fastq = write_fastq(tmp_path / "reads.fastq.gz", [
        ACTUAL_ADAPTER + GUIDE,
        ACTUAL_ADAPTER + GUIDE,
        ACTUAL_ADAPTER + UNIQUE,
        "ACGT" * 10,
    ])
    result = rs.count_guides(
        fastq,
        reference_frame(),
        mode="count-flex",
        sample="sample",
        orientation="forward",
        adapter=ADAPTER,
    )

    assert isinstance(result, rs.FlexibleCountResult)
    assert result.counts_df.to_dict("records") == [
        {"sgRNA": "unique", "Gene": "GENE3", "sample": 1}
    ]
    assert int(result.sequence_counts_df["count"].sum()) == 3
    duplicate_rows = result.guide_audit_df[result.guide_audit_df.sgRNA != "unique"]
    assert duplicate_rows["resolved_count"].isna().all()
    assert duplicate_rows["compatible_group_reads"].tolist() == [2, 2]
    assert result.gene_counts_df.to_dict("records") == [{"gene": "GENE3", "count": 1}]
    assert result.qc_summary["total_reads"] == 4
    assert result.qc_summary["matched_sequence_reads"] == 3
    assert result.rejection_stats == {
        "assigned_sequence_group": 2,
        "assigned_unique_id": 1,
        "adapter_not_found": 1,
    }
    counts, qc, rejected = result
    assert counts is result.counts_df and qc is result.qc_summary
    assert rejected is result.rejection_stats


def test_count_flex_api_and_cli_tables_are_equivalent(tmp_path):
    reference = reference_frame()
    reference_path = tmp_path / "reference.tsv"
    reference.to_csv(reference_path, sep="\t", index=False)
    fastq = write_fastq(tmp_path / "reads.fastq", [
        ACTUAL_ADAPTER + GUIDE,
        ACTUAL_ADAPTER + UNIQUE,
        ACTUAL_ADAPTER + UNIQUE,
    ])
    api = rs.count_guides(
        fastq,
        reference_path,
        mode="count-flex",
        sample="S",
        orientation="forward",
        adapter=ADAPTER,
    )
    output = tmp_path / "cli"
    assert main([
        "count-flex", "--library", str(reference_path), "--fastq", str(fastq),
        "--sample", "S", "--orientation", "forward", "--adapter", ADAPTER,
        "--output", str(output),
    ]) == 0

    pd.testing.assert_frame_equal(
        api.sequence_counts_df,
        pd.read_csv(output / "sequence_counts.tsv", sep="\t"),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        api.guide_audit_df,
        pd.read_csv(output / "guide_audit.tsv", sep="\t"),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        api.counts_df,
        pd.read_csv(output / "resolved_counts.tsv", sep="\t"),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        api.gene_counts_df,
        pd.read_csv(output / "identifiable_gene_counts.tsv", sep="\t"),
        check_dtype=False,
    )


def test_count_guides_anchor_mode_dispatches_to_existing_api(tmp_path):
    reference = pd.DataFrame({"sgRNA": ["g"], "Gene": ["GENE"], "Sequence": [GUIDE]})
    fastq = write_fastq(tmp_path / "reads.fastq", ["CACCG" + GUIDE + "GTTTT"])
    direct = rs.count(fastq, reference, orientation="forward")
    dispatched = rs.count_guides(fastq, reference, mode="count", orientation="forward")
    pd.testing.assert_frame_equal(direct.counts_df, dispatched.counts_df)
    assert direct.rejection_stats == dispatched.rejection_stats


@pytest.mark.parametrize("kwargs,exception", [
    ({"mode": "unknown", "orientation": "forward", "adapter": ADAPTER}, ValueError),
    ({"mode": "count-flex", "orientation": "auto", "adapter": ADAPTER}, ValueError),
    ({"mode": "count-flex", "orientation": "forward"}, ValueError),
    ({"mode": "count-flex", "orientation": "forward", "adapter": 123}, TypeError),
    ({"mode": "count-flex", "orientation": "forward", "adapter": ADAPTER,
      "max_adapter_edits": 1.5}, ValueError),
    ({"mode": "count-flex", "orientation": "forward", "adapter": ADAPTER,
      "max_adapter_edits": True}, ValueError),
    ({"mode": "count-flex", "orientation": "forward", "adapter": ADAPTER,
      "left_anchor": "CACCG"}, TypeError),
])
def test_count_guides_rejects_ambiguous_options(tmp_path, kwargs, exception):
    fastq = write_fastq(tmp_path / "reads.fastq", [ACTUAL_ADAPTER + GUIDE])
    with pytest.raises(exception):
        rs.count_guides(fastq, reference_frame(), **kwargs)
