import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from readsparrow.cli import main
from readsparrow.empirical_inference import (EmpiricalConfig, call_empirical_hits,
    predictive_background, signed_rank_p, split_controls)
from readsparrow.io import CountTable, save_counts, write_tsv
from readsparrow.simulation import SimulationConfig, simulate
from readsparrow.null_comparison import comparison_suite, generate


class EmpiricalTests(unittest.TestCase):
    def setUp(self):
        self.table, self.meta, self.truth, _ = simulate(SimulationConfig(genes=100, controls=200, seed=197))

    def test_ranks_ties_floors_and_no_zero_p(self):
        pp, pn = signed_rank_p([0, 1, 1, 2], [-100, 1, 100])
        np.testing.assert_allclose(pp, [1, .8, .2])
        np.testing.assert_allclose(pn, [.2, .8, 1])
        with self.assertRaises(ValueError):
            signed_rank_p([], [0])

    def test_split_is_balanced_disjoint_and_row_order_invariant(self):
        ids, mask = ["a", "b", "c", "d", "e"], np.ones(5, bool)
        a, b = split_controls(ids, mask, "salt")
        c, d = split_controls(ids[::-1], mask, "salt")
        self.assertFalse(set(a) & set(b))
        self.assertEqual([ids[i] for i in a], [ids[::-1][i] for i in c])
        self.assertEqual(len(a), 2)
        self.assertEqual(len(b), 3)

    def test_new_models_do_not_delete_guides_and_empirical_floor(self):
        for model in ("background", "empirical"):
            rows, info, guides = call_empirical_hits(self.table, self.meta, "treatment", "control", EmpiricalConfig(null_model=model))
            self.assertEqual(info["background_model"]["training_controls"], 100)
            self.assertEqual(info["background_model"]["calibration_controls"], 100)
            self.assertFalse(info["background_model"]["fitted_from_target_effects"])
            self.assertTrue(all(not r["removed_from_test"] for r in guides))
            self.assertEqual(info["residual_df_per_guide"], 4)
            if model == "empirical":
                self.assertTrue(all(r["p_enrichment"] >= 1 / 101 for r in guides))
                # m=4 simultaneous guide intervals require tails .00625, below
                # resolution 1/101, so no finite gene interval can be justified.
                self.assertTrue(all(not r["ci_informative"] for r in rows))

    def test_missing_controls_is_error_not_silent_fallback(self):
        t, m, _, _ = simulate(SimulationConfig(genes=50, controls=99))
        with self.assertRaisesRegex(ValueError, "no silent"):
            call_empirical_hits(t, m, "treatment", "control")

    def test_training_moments_do_not_use_target_effects(self):
        rng = np.random.default_rng(112)
        beta, se, abundance = rng.normal(size=1000), np.full(1000, .2), rng.normal(size=1000)
        train = np.arange(100)
        a = predictive_background(beta, se, abundance, train, EmpiricalConfig(), 50)
        beta[100:] *= 100
        b = predictive_background(beta, se, abundance, train, EmpiricalConfig(), 50)
        for left, right in zip(a[:4], b[:4]):
            np.testing.assert_array_equal(left, right)
        self.assertTrue(np.all(a[1] >= se))

    def test_low_information_keeps_p_one_and_required_support(self):
        x = self.table.counts.copy()
        x[0] = 0
        t = CountTable(self.table.ids, self.table.genes, self.table.samples, x)
        rows, _, guides = call_empirical_hits(t, self.meta, "treatment", "control")
        self.assertEqual(rows[0]["guides"], 4)
        self.assertEqual(rows[0]["required_support"], 2)
        self.assertEqual(guides[0]["p_enrichment"], 1)
        self.assertIsNone(guides[0]["standard_error"])

    def test_row_order_invariance(self):
        a, ai, _ = call_empirical_hits(self.table, self.meta, "treatment", "control")
        order = np.random.default_rng(107).permutation(len(self.table.ids))
        t = CountTable([self.table.ids[i] for i in order], [self.table.genes[i] for i in order], self.table.samples, self.table.counts[order])
        b, bi, _ = call_empirical_hits(t, self.meta, "treatment", "control")
        np.testing.assert_allclose([r["pvalue"] for r in a], [r["pvalue"] for r in b], rtol=1e-8)
        self.assertEqual(ai["background_model"]["training_control_ids"], bi["background_model"]["training_control_ids"])

    def test_cli_new_models_are_standalone_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            save_counts(p / "counts.tsv", self.table)
            write_tsv(p / "samples.tsv", list(self.meta[0]), self.meta)
            args = ["test", "--counts", str(p / "counts.tsv"), "--samples", str(p / "samples.tsv"),
                    "--treatment", "treatment", "--control", "control", "--experimental", "--output", str(p / "out")]
            with patch("subprocess.run", side_effect=AssertionError("External inference forbidden")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            info = json.loads((p / "out/analysis.json").read_text())
            self.assertEqual(info["config"]["null_model"], "background")
            self.assertTrue((p / "out/null_control_audit.tsv").exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)

    def test_invalid_configuration(self):
        for cfg in (EmpiricalConfig(min_null_controls=99), EmpiricalConfig(null_model="magic"),
                    EmpiricalConfig(min_training_per_bin=1), EmpiricalConfig(max_background_bins=0)):
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_cached_and_uncached_models_are_exactly_identical(self):
        for guides, paired, replicates in ((2, False, 2), (4, False, 3), (6, True, 3)):
            table, meta, _, _ = simulate(SimulationConfig(genes=100, controls=200, guides_per_gene=guides,
                                                         paired=paired, replicates=replicates, seed=8917 + guides))
            for method in ("background", "empirical"):
                config = EmpiricalConfig(null_model=method)
                cached = call_empirical_hits(table, meta, "treatment", "control", config)
                with patch("readsparrow.empirical_inference.lru_cache", side_effect=lambda **kw: lambda fn: fn):
                    uncached = call_empirical_hits(table, meta, "treatment", "control", config)
                self.assertEqual(cached, uncached)

    def test_interval_quantiles_are_cached_by_guide_family(self):
        from scipy.stats import t
        original = t.ppf
        with patch("readsparrow.empirical_inference.student_t.ppf", wraps=original) as counter:
            call_empirical_hits(self.table, self.meta, "treatment", "control")
        # One call for the legacy per-family rectangle and one for the new
        # cached prediction rectangle, not a full-array PPF per target gene.
        self.assertLessEqual(counter.call_count, 4)

    def test_drift_generator_is_seeded_and_retains_biological_truth(self):
        cfg = SimulationConfig(genes=100, controls=200, active_fraction=0, seed=128)
        a, _, truth = generate(cfg, drift="gaussian")
        b, _, _ = generate(cfg, drift="gaussian")
        np.testing.assert_array_equal(a.counts, b.counts)
        self.assertTrue(all(not row["active"] for row in truth))
        self.assertTrue(np.all(a.counts >= 0))

    def test_low_count_refusal_is_not_scored_as_zero_fdr(self):
        with tempfile.TemporaryDirectory() as td:
            result = comparison_suite(Path(td), repeats=2, null_repeats=2, genes=100,
                                      scenario_names=["low_counts"])
            new = [r for r in result["scenarios"] if r["method"] in {"background", "empirical"}]
            self.assertTrue(all(r["valid_repetitions"] == 0 for r in new))
            self.assertTrue(all("fdp" not in r for r in new))


if __name__ == "__main__":
    unittest.main()
