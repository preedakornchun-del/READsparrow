import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from readsparrow.cli import main, parser
from readsparrow.correlated_comparison import cases, generate
from readsparrow.empirical_inference import EmpiricalConfig, call_empirical_hits
from readsparrow.io import CountTable, save_counts, write_tsv
from readsparrow.shape_inference import ShapeConfig, call_shape_hits, fit_shape, sas_latent, sas_nll, subset_scan, score_shape
from readsparrow.sparse_posterior import posterior_decisions, sparse_log_bayes_factor, three_class_decisions


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.case = cases()["matched_ineffective_rho0.5"]
        self.cfg = replace(self.case["config"], genes=100, controls=200, seed=281023)
        self.table, self.meta, self.truth, self.blocks = generate(self.cfg, self.case)

    def test_normal_nested_exactly(self):
        x = np.array([-4., -1., 0., 1., 4.])
        np.testing.assert_allclose(sas_latent(x, [0, 0, 0, 0]), x)
        self.assertAlmostEqual(sas_nll(np.array([0, 0, 0, 0]), x), -norm.logpdf(x).sum())

    def test_shape_monotone_finite(self):
        z = sas_latent(np.linspace(-50, 50, 1000), [0., .1, .8, -.3])
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue((np.diff(z) > 0).all())

    def test_fit_skew_uses_no_target(self):
        x = np.random.default_rng(93).lognormal(0, .7, 2000)
        model = fit_shape(x)
        self.assertEqual(model["family"], "sinh-arcsinh")
        self.assertLess(model["bic_shape"], model["bic_gaussian"])

    def test_invalid_shape_config(self):
        for config in [ShapeConfig(tail_inflation=.5), ShapeConfig(scan_weight=1), ShapeConfig(max_scan_guides=9), ShapeConfig(matched_inference="fake")]:
            with self.assertRaises(ValueError): config.validate()

    def test_posterior_values_not_pvalues(self):
        rows, info, _ = call_shape_hits(self.table, self.meta, "treatment", "control", ShapeConfig(correlation_mode="matched-blocks"), control_blocks=self.blocks)
        tested = [r for r in rows if r["tested"]]
        self.assertTrue(all(r["pvalue"] is None for r in rows))
        self.assertTrue(all((min(1., 2*r["posterior_qvalue"]) if info["posterior_model"]["global_gate_pass"] else 1.) == r["qvalue"] for r in tested))
        self.assertIn("NOT BH", info["fdr_family"])
        self.assertIsNone(score_shape(rows, self.truth, 4)["null_p_lt_005"])

    def test_no_matched_groups_no_posterior(self):
        rows, info, _ = call_shape_hits(self.table, self.meta, "treatment", "control")
        self.assertIsNone(info["posterior_model"])
        self.assertTrue(all(r["pvalue"] is not None for r in rows if r["tested"]))
        self.assertTrue(all(r["ci_low"] is None and r["ci_high"] is None for r in rows))

    def test_no_silent_matched_fallback(self):
        with self.assertRaisesRegex(ValueError, "requires genuine"):
            call_shape_hits(self.table, self.meta, "treatment", "control", ShapeConfig(correlation_mode="matched-blocks"))

    def test_posterior_q_arithmetic(self):
        logs = np.r_[np.repeat(-4., 80), np.repeat(10., 20)]
        local, q, info = posterior_decisions(logs)
        order = np.argsort(local, kind="stable")
        np.testing.assert_allclose(q[order], np.cumsum(local[order])/np.arange(1, 101))
        self.assertLessEqual(info["estimated_alternative_prevalence"], .5)

    def test_no_evidence_prevalence_zero(self):
        local, q, info = posterior_decisions(np.full(100, -10.))
        self.assertEqual(info["estimated_alternative_prevalence"], 0.)
        np.testing.assert_equal(q, np.ones(100))

    def test_three_class_clean_boundary_no_global_signal(self):
        local, q, info = three_class_decisions(np.full(100,-5.), np.full(100,-10.))
        self.assertFalse(info["global_gate_pass"])
        self.assertEqual(info["estimated_alternative_prevalence"],0.)

    def test_three_class_global_signal_and_probabilities(self):
        single = np.r_[np.full(80,-5.),np.full(20,0.)]
        multi = np.r_[np.full(80,-5.),np.full(20,12.)]
        local,q,info=three_class_decisions(single,multi)
        self.assertTrue(info["global_gate_pass"])
        self.assertTrue(np.all((local>=0)&(local<=1)))
        self.assertLess(q[-1],.05)

    def test_single_class_not_called_multi(self):
        single=np.r_[np.full(80,-3.),np.full(20,12.)]
        multi=np.full(100,-5.)
        _,_,info=three_class_decisions(single,multi)
        self.assertFalse(info["global_gate_pass"])

    def test_skew_bic_gate_disables_posterior(self):
        model=dict(family="sinh-arcsinh",parameters=[0.,0.,.5,0.],training_controls=100)
        with patch("readsparrow.shape_inference.fit_shape",return_value=model):
            rows,info,_=call_shape_hits(self.table,self.meta,"treatment","control",ShapeConfig(correlation_mode="matched-blocks"),control_blocks=self.blocks)
        self.assertIsNone(info["posterior_model"])
        self.assertEqual(info["inference_branch"],"guarded-shape-marginal-BH")
        self.assertTrue(all(r["pvalue"] is not None for r in rows if r["tested"]))

    def test_low_information_gene_not_deleted_to_increase_power(self):
        x=self.table.counts.copy();x[0]=0
        table=CountTable(self.table.ids,self.table.genes,self.table.samples,x)
        rows,_,_=call_shape_hits(table,self.meta,"treatment","control")
        self.assertFalse(rows[0]["tested"])

    def test_single_extreme_artifact_competes_with_alternative(self):
        values = np.array([[20., 0., 0., 0.], [4., 4., 0., 0.]])
        bf = sparse_log_bayes_factor(values, .5, 1.)
        self.assertLess(bf[0], bf[1])

    def test_bayes_factor_permutation_and_sign_invariance(self):
        values = np.random.default_rng(5).normal(size=(100, 4))
        before = sparse_log_bayes_factor(values, .5, 1.)
        np.testing.assert_allclose(before, sparse_log_bayes_factor(values[:, [3, 0, 2, 1]], .5, 1.), atol=1e-12)
        np.testing.assert_allclose(before, sparse_log_bayes_factor(-values, .5, 1.), atol=1e-12)

    def test_marginal_guard_no_target_scale_shrink(self):
        rows, info, guides = call_shape_hits(self.table, self.meta, "treatment", "control")
        self.assertTrue(all(r["shape_p_enrichment"] >= 0 and r["shape_p_depletion"] >= 0 for r in guides))
        self.assertTrue(all(r["shape_p_enrichment"] + r["shape_p_depletion"] >= 1-1e-12 for r in guides))

    def test_prepared_context_exact(self):
        context = {}
        rows, info, guides = call_empirical_hits(self.table, self.meta, "treatment", "control", EmpiricalConfig(), _context_sink=context)
        direct = call_shape_hits(self.table, self.meta, "treatment", "control")
        prepared = call_shape_hits(self.table, self.meta, "treatment", "control", _prepared=(rows, info, guides, context))
        np.testing.assert_equal([r["pvalue"] for r in direct[0]], [r["pvalue"] for r in prepared[0]])

    def test_row_order_invariance(self):
        config = ShapeConfig(correlation_mode="matched-blocks")
        before, _, _ = call_shape_hits(self.table, self.meta, "treatment", "control", config, control_blocks=self.blocks)
        order = np.random.default_rng(7).permutation(len(self.table.ids))
        perm = CountTable([self.table.ids[i] for i in order], [self.table.genes[i] for i in order], self.table.samples, self.table.counts[order])
        after, _, _ = call_shape_hits(perm, self.meta, "treatment", "control", config, control_blocks=self.blocks)
        np.testing.assert_allclose([r["qvalue"] for r in before if r["tested"]], [r["qvalue"] for r in after if r["tested"]], atol=1e-10)

    def test_sample_order_invariance(self):
        before, _, _ = call_shape_hits(self.table, self.meta, "treatment", "control")
        order = [5, 1, 3, 4, 0, 2]
        perm = CountTable(self.table.ids, self.table.genes, [self.table.samples[i] for i in order], self.table.counts[:, order])
        after, _, _ = call_shape_hits(perm, self.meta, "treatment", "control")
        np.testing.assert_allclose([r["qvalue"] for r in before if r["tested"]], [r["qvalue"] for r in after if r["tested"]], atol=1e-7)

    def test_cli_no_external_engine_and_default_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_counts(root/"counts.tsv", self.table)
            write_tsv(root/"samples.tsv", list(self.meta[0]), self.meta)
            write_tsv(root/"blocks.tsv", ["sgRNA", "block"], [dict(sgRNA=g, block=b) for g, b in self.blocks.items()])
            with patch("subprocess.run", side_effect=AssertionError("No external engine")), contextlib.redirect_stdout(io.StringIO()):
                code = main(["test", "--counts", str(root/"counts.tsv"), "--samples", str(root/"samples.tsv"),
                    "--treatment", "treatment", "--control", "control", "--method", "shape-adaptive", "--control-blocks", str(root/"blocks.tsv"),
                    "--experimental", "--output", str(root/"output")])
            self.assertEqual(code, 0)
            report = json.loads((root/"output/analysis.json").read_text())
            self.assertIn("NOT BH", report["fdr_family"])
            self.assertEqual(parser().parse_args(["test", "--counts", "a", "--samples", "b", "--treatment", "t", "--control", "c", "--output", "o"]).method, "background")


if __name__ == "__main__": unittest.main()
