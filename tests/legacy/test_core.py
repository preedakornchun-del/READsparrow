import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from readsparrow.calibration import binomial_interval, evaluate
from readsparrow.counting import ExactExtractor, Library, count_fastq, fastq_records, load_library, reverse_complement
from readsparrow.inference import TestConfig, bh_adjust, call_hits, fit_log_model, make_design, moderate_variances, size_factors
from readsparrow.io import CountTable, integer_count, load_counts, new_output_dir, save_counts, write_tsv
from readsparrow.qc import gini
from readsparrow.simulation import SimulationConfig, simulate


class InputTests(unittest.TestCase):
    def test_count_parser_rejects_lossy_values(self):
        for text in ("1.9", "-1", "NaN", "inf", "", "1e30", "9223372036854775808"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                integer_count(text)
        self.assertEqual(integer_count("9007199254740993"), 9007199254740993)
        self.assertEqual(integer_count("2.0"), 2)

    def test_table_roundtrip(self):
        table, _, _, _ = simulate(SimulationConfig(genes=50))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "counts.tsv"
            save_counts(path, table)
            result = load_counts(path)
            np.testing.assert_array_equal(table.counts, result.counts)
            self.assertEqual(table.ids, result.ids)
            self.assertEqual(table.genes, result.genes)
            self.assertEqual(table.samples, result.samples)

    def test_duplicate_and_overflow_counts(self):
        for ids, x in [(["x", "x"], np.ones((2, 1), dtype=int)), (["x", "y"], np.array([[2**63 - 1], [1]])),
                       (["x", "y"], np.ones((2, 1))), (["x", "y"], np.array([[1], [-1]]))]:
            with self.assertRaises(ValueError):
                CountTable(ids, ["G", "G"], ["s"], x)

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = new_output_dir(Path(temp) / "out")
            (path / "user.txt").write_text("keep")
            with self.assertRaises(ValueError):
                new_output_dir(path)
            self.assertEqual((path / "user.txt").read_text(), "keep")

    def test_library_validation(self):
        for sequences in ([b"AAAA", b"AAAA"], [b"AAAA", b"AAC"], [b"AAAA", b"NNNN"]):
            with self.assertRaises(ValueError):
                Library(["a", "b"], ["G1", "G2"], sequences)


class CountingTests(unittest.TestCase):
    def test_forward_and_reverse_19_20(self):
        for length in (19, 20):
            guide = b"A" * (length - 4) + b"CGTA"
            lib = Library(["s"], ["G"], [guide])
            seq = b"ACACACAC" + b"CACCG" + guide + b"GTTTT" + b"CC"
            self.assertEqual(ExactExtractor(lib).match(seq), (0, "assigned_exact"))
            self.assertEqual(ExactExtractor(lib, "reverse").match(reverse_complement(seq)), (0, "assigned_exact"))

    def test_internal_downstream_anchor_protected(self):
        guide = b"AAAGTTTT" + b"C" * 12
        ext = ExactExtractor(Library(["s"], ["G"], [guide]))
        self.assertEqual(ext.match(b"CACCG" + guide + b"GTTTT")[0], 0)

    def test_position_shift_and_anchor_mutation(self):
        guide = b"ACGT" * 5
        ext = ExactExtractor(Library(["s"], ["G"], [guide]))
        self.assertEqual(ext.match(b"CCACCGAA" + guide + b"GTTTT")[0], 0)
        self.assertEqual(ext.match(b"CATCG" + guide + b"GTTTT")[1], "no_left_anchor")
        self.assertEqual(ext.match(b"CACCG" + guide + b"GTTTA")[1], "no_usable_right_anchor")

    def test_first_match_legacy_rule_explicit(self):
        a, b = b"A" * 20, b"C" * 20
        ext = ExactExtractor(Library(["a", "b"], ["G1", "G2"], [a, b]))
        self.assertEqual(ext.match(b"CACCG" + a + b + b"GTTTT")[0], 0)

    def test_unique_policy_rejects_competing_guides(self):
        a, b = b"A" * 20, b"C" * 20
        lib = Library(["a", "b"], ["G1", "G2"], [a, b])
        ext = ExactExtractor(lib, policy="unique")
        self.assertEqual(ext.match(b"CACCG" + a + b + b"GTTTT"), (None, "ambiguous_multiple_guides"))
        self.assertEqual(ext.match(b"CACCG" + a + b"GTTTT" + b"CACCG" + a + b"GTTTT")[0], 0)

    def test_unique_policy_handles_multiple_anchor_occurrences(self):
        guide = b"ACGT" * 5
        ext = ExactExtractor(Library(["a"], ["G"], [guide]), policy="unique", max_extra=0)
        self.assertEqual(ext.match(b"CACCG" + b"A" * 50 + b"CACCG" + guide + b"GTTTT")[0], 0)
        self.assertEqual(ext.match(b"CACCGAA" + guide + b"GTTTT")[1], "anchor_span_exceeds_limit")

    def test_custom_anchor_and_variable_prefix(self):
        guide = b"ACGT" * 5
        ext = ExactExtractor(Library(["a"], ["G"], [guide]), left=b"AGAGA", right=b"CTCTC", policy="unique")
        for n in (0, 1, 10, 51):
            self.assertEqual(ext.match(b"A" * n + b"AGAGA" + guide + b"CTCTC")[0], 0)

    def test_fastq_validation_and_gzip(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "reads.fastq.gz"
            sequence = b"CACCG" + b"A" * 20 + b"GTTTT"
            with gzip.open(p, "wb") as f:
                f.write(b"@r\n" + sequence + b"\n+\n" + b"I" * len(sequence) + b"\n")
            lib = Library(["a"], ["G"], [b"A" * 20])
            counts, metrics = count_fastq(lib, p)
            self.assertEqual(int(counts.counts.sum()), 1)
            self.assertEqual(metrics["total_reads"], 1)
            p = Path(temp) / "bad.fastq"
            for content in (b"@r\nACGT\n+\n", b"@r\nACGT\n+\nIII\n", b"no\nACGT\n+\nIIII\n"):
                p.write_bytes(content)
                with self.assertRaises(ValueError):
                    list(fastq_records(p))

    def test_qc_known_gini(self):
        self.assertEqual(gini([0, 0]), 0)
        self.assertEqual(gini([1, 1]), 0)
        self.assertAlmostEqual(gini([0, 2]), .5)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.table, self.metadata, self.truth, _ = simulate(SimulationConfig(genes=300, seed=144))

    def test_bh_known_example(self):
        np.testing.assert_allclose(bh_adjust([.01, .04, .03, .002]), [.02, .04, .04, .008])
        self.assertEqual(bh_adjust([]).size, 0)
        for p in ([np.nan], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                bh_adjust(p)

    def test_normalization_scale_invariance(self):
        a, _ = size_factors(self.table.counts)
        b, _ = size_factors(self.table.counts * 7)
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(np.prod(a)), 1)

    def test_no_normalization_fallback(self):
        with self.assertRaises(ValueError):
            size_factors(np.array([[0, 2], [3, 0]]))
        with self.assertRaises(ValueError):
            size_factors(np.zeros((30, 2)))
        with self.assertRaises(ValueError):
            size_factors(np.ones((30, 2)), "control", np.zeros(30, dtype=bool))

    def test_reverse_contrast(self):
        a, ai = call_hits(self.table, self.metadata, "treatment", "control")
        b, bi = call_hits(self.table, self.metadata, "control", "treatment")
        for left, right in zip(a, b):
            if left["tested"]:
                self.assertAlmostEqual(left["log2_effect"], -right["log2_effect"], places=10)
                self.assertAlmostEqual(left["pvalue"], right["pvalue"], places=10)
                self.assertAlmostEqual(left["ci_low"], -right["ci_high"], places=10)
        self.assertEqual(ai["residual_df"], 4)
        self.assertEqual(ai["biological_samples"], 6)

    def test_sample_order_invariance(self):
        a, _ = call_hits(self.table, self.metadata, "treatment", "control")
        order = [5, 1, 3, 0, 2, 4]
        table = CountTable(self.table.ids, self.table.genes, [self.table.samples[i] for i in order], self.table.counts[:, order])
        b, _ = call_hits(table, self.metadata, "treatment", "control")
        np.testing.assert_allclose([r["pvalue"] for r in a], [r["pvalue"] for r in b], rtol=1e-9)

    def test_guides_are_not_biological_replicates(self):
        duplicated = CountTable(self.table.ids + [s + "copy" for s in self.table.ids], self.table.genes * 2,
                                self.table.samples, np.vstack([self.table.counts, self.table.counts]))
        _, a = call_hits(self.table, self.metadata, "treatment", "control")
        _, b = call_hits(duplicated, self.metadata, "treatment", "control")
        self.assertEqual(a["residual_df"], b["residual_df"])
        self.assertEqual(a["biological_samples"], b["biological_samples"])

    def test_missing_bio_ids_and_technical_replicates_rejected(self):
        for missing in (True, False):
            metadata = [dict(r) for r in self.metadata]
            if missing:
                metadata[0].pop("biological_replicate")
            else:
                metadata[1]["biological_replicate"] = metadata[0]["biological_replicate"]
            with self.assertRaises(ValueError):
                call_hits(self.table, metadata, "treatment", "control")

    def test_batch_confounding_rejected(self):
        metadata = [{**r, "batch": r["condition"]} for r in self.metadata]
        with self.assertRaises(ValueError):
            call_hits(self.table, metadata, "treatment", "control")

    def test_paired_design_and_two_pairs_rejected(self):
        table, metadata, _, _ = simulate(SimulationConfig(genes=100, paired=True, replicates=3))
        _, info = call_hits(table, metadata, "treatment", "control")
        self.assertEqual(info["residual_df"], 2)
        table, metadata, _, _ = simulate(SimulationConfig(genes=100, paired=True, replicates=2))
        with self.assertRaises(ValueError):
            call_hits(table, metadata, "treatment", "control")

    def test_filtered_rows_never_become_hits(self):
        counts = self.table.counts.copy()
        counts[:4] = 0
        table = CountTable(self.table.ids, self.table.genes, self.table.samples, counts)
        rows, info = call_hits(table, self.metadata, "treatment", "control")
        self.assertFalse(rows[0]["tested"])
        self.assertIsNone(rows[0]["qvalue"])
        self.assertFalse(rows[0]["hit"])
        self.assertEqual(info["control_guides_excluded"], 100)
        self.assertNotIn("NA", [r["gene"] for r in rows])

    def test_baseline_absence_and_guide_dominance_are_flags_not_corrections(self):
        counts = self.table.counts.copy()
        counts[:4, :3] = 0
        counts[0, 3:] = 10000
        rows, info = call_hits(CountTable(self.table.ids, self.table.genes, self.table.samples, counts),
                               self.metadata, "treatment", "control")
        self.assertIn("absent_at_baseline", rows[0]["diagnostic_flags"])
        self.assertIn("guide_dominance", rows[0]["diagnostic_flags"])
        self.assertEqual(rows[0]["baseline_zero_guides"], 4)
        self.assertIn("normalization_audit", info)

    def test_focused_library_control_normalization_not_automatically_selected(self):
        table, meta, _, _ = simulate(SimulationConfig(genes=400, active_fraction=.8, positive_fraction=1, seed=881))
        _, info = call_hits(table, meta, "treatment", "control")
        audit = info["normalization_audit"]
        self.assertFalse(audit["automatic_method_selection"])
        self.assertEqual(audit["chosen_method"], "median")
        self.assertGreater(audit["alternatives"]["control"]["max_absolute_log2_factor_difference"], .5)

    def test_ols_matches_closed_form(self):
        rng = np.random.default_rng(144)
        y = rng.normal(size=(500, 8))
        x = np.column_stack([np.ones(8), [0] * 4 + [1] * 4])
        effect, se, p, df, _ = fit_log_model(y, x)
        np.testing.assert_allclose(effect, y[:, 4:].mean(axis=1) - y[:, :4].mean(axis=1), atol=1e-12)
        np.testing.assert_allclose(p, 2 * student_t.sf(np.abs(effect / se), df))

    def test_scaled_f_hyperparameters(self):
        rng = np.random.default_rng(781)
        n, df, df0, scale = 40000, 4, 10, .3
        variances = scale * (rng.chisquare(df, n) / df) / (rng.chisquare(df0, n) / df0)
        posterior, info = moderate_variances(variances, np.zeros(n), df)
        self.assertLess(abs(info["prior_df"] - df0), 1)
        prior_estimate = (posterior * (df + info["prior_df"]) - df * variances) / info["prior_df"]
        self.assertLess(abs(np.median(prior_estimate) - scale), .02)

    def test_too_few_genes_and_invalid_parameters(self):
        with self.assertRaises(ValueError):
            moderate_variances(np.ones(10), np.zeros(10), 4)
        for config in (TestConfig(fdr=1), TestConfig(pseudocount=0), TestConfig(min_count=0)):
            with self.assertRaises(ValueError):
                call_hits(self.table, self.metadata, "treatment", "control", config)


class SimulationTests(unittest.TestCase):
    def test_exact_all_null_truth(self):
        for seed in range(10):
            _, _, truth, _ = simulate(SimulationConfig(seed=seed, genes=100, active_fraction=0))
            self.assertEqual(sum(r["active"] for r in truth), 0)
            self.assertTrue(all(r["true_log2_effect"] == 0 for r in truth))

    def test_seed_reproducibility_and_effect_truth(self):
        config = SimulationConfig(genes=100, seed=35)
        a, _, truth, _ = simulate(config)
        b, _, _, _ = simulate(config)
        np.testing.assert_array_equal(a.counts, b.counts)
        self.assertEqual(sum(r["active"] for r in truth), 10)
        self.assertTrue(all(abs(r["true_log2_effect"] - r["nominal_effect"]) < 1e-12 for r in truth))
        c, _, _, _ = simulate(replace(config, seed=36))
        self.assertFalse(np.array_equal(a.counts, c.counts))

    def test_all_null_fdr_definition(self):
        truth = [{"gene": "G", "active": False, "true_log2_effect": 0}]
        r = {"gene": "G", "tested": True, "pvalue": .001, "qvalue": .001, "log2_effect": 2., "ci_low": 1., "ci_high": 3.}
        metric = evaluate([r], truth)
        self.assertEqual(metric["fdp"], 1)
        self.assertEqual(metric["any_hit"], 1)
        self.assertEqual(metric["false_hits"], 1)
        r["qvalue"] = 1
        self.assertEqual(evaluate([r], truth)["fdp"], 0)
        interval = binomial_interval(0, 100)
        self.assertEqual(interval[0], 0)
        self.assertGreater(interval[1], 0)

    def test_artifact_truth_remains_biologically_null(self):
        _, _, truth, _ = simulate(SimulationConfig(genes=100, active_fraction=0, artifact_fraction=.1))
        self.assertEqual(sum(r["technical_artifact"] for r in truth), 10)
        self.assertEqual(sum(r["active"] for r in truth), 0)


if __name__ == "__main__":
    unittest.main()
