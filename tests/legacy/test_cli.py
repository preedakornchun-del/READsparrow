import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from readsparrow.cli import main
from readsparrow.io import load_counts, write_tsv


class CLITests(unittest.TestCase):
    def test_fastq_to_hits_without_subprocess_or_mageck(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(713)
            library = [{"sgRNA": f"sg_{i}", "Gene": f"G_{i // 2:03d}",
                        "Sequence": "".join(rng.choice(list("ACGT"), size=20))} for i in range(80)]
            write_tsv(root / "library.tsv", list(library[0]), library)
            samples = []
            for j in range(6):
                sample, condition = f"sample{j}", "control" if j < 3 else "treatment"
                samples.append({"sample": sample, "condition": condition, "biological_replicate": f"bio{j}",
                                "fastq": f"{sample}.fastq", "orientation": "forward", "policy": "unique"})
                with (root / f"{sample}.fastq").open("w") as handle:
                    for i, r in enumerate(library):
                        repeats = 5 + j % 3 + (8 if i < 8 and j >= 3 else 0)
                        seq = "ACAC" + "CACCG" + r["Sequence"] + "GTTTT" + "CACA"
                        for k in range(repeats):
                            handle.write(f"@{i}_{k}\n{seq}\n+\n{'I' * len(seq)}\n")
            write_tsv(root / "samples.tsv", list(samples[0]), samples)
            # Any hidden R/MAGeCK/external process invocation would fail this test.
            with patch("subprocess.run", side_effect=AssertionError("No external engine allowed")), contextlib.redirect_stdout(io.StringIO()):
                code = main(["run", "--library", str(root / "library.tsv"), "--samples", str(root / "samples.tsv"),
                             "--treatment", "treatment", "--control", "control", "--method", "guide-aware", "--experimental", "--output", str(root / "run")])
            self.assertEqual(code, 0)
            counts = load_counts(root / "run/counts.tsv")
            self.assertEqual(counts.counts.shape, (80, 6))
            info = json.loads((root / "run/analysis.json").read_text())
            self.assertEqual(info["tested_genes"], 40)
            self.assertEqual(info["residual_df_per_guide"], 4)
            provenance = json.loads((root / "run/provenance.json").read_text())
            self.assertFalse(provenance["MAGeCK_required"])
            self.assertTrue((root / "run/report.html").exists())

    def test_experimental_acknowledgement_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["test", "--counts", "unused", "--samples", "unused", "--treatment", "T", "--control", "C", "--output", "unused"])
        self.assertEqual(caught.exception.code, 2)

    def test_standalone_count_unique_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_tsv(root / "lib.tsv", ["sgRNA", "Gene", "Sequence"], [{"sgRNA": "a", "Gene": "G", "Sequence": "ACGT" * 5}])
            seq = "CACCG" + "ACGT" * 5 + "GTTTT"
            (root / "r.fastq").write_text(f"@a\n{seq}\n+\n{'I' * len(seq)}\n")
            self.assertEqual(main(["count", "--library", str(root / "lib.tsv"), "--fastq", str(root / "r.fastq"),
                                   "--sample", "S", "--orientation", "forward", "--policy", "unique", "--output", str(root / "out")]), 0)
            self.assertEqual(load_counts(root / "out/counts.tsv").counts.sum(), 1)


if __name__ == "__main__":
    unittest.main()
