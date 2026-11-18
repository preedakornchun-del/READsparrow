import unittest
from dataclasses import replace

import numpy as np

from readsparrow.guide_inference import GuideConfig, adjust_pvalues, call_guide_hits, partial_conjunction
from readsparrow.io import CountTable
from readsparrow.simulation import SimulationConfig, simulate


class PartialConjunctionTests(unittest.TestCase):
    def test_known_values_and_arbitrary_single_outlier(self):
        self.assertAlmostEqual(partial_conjunction([1e-100, .2, .5, .8], 2), .6)
        self.assertAlmostEqual(partial_conjunction([1e-100, .0001, .5, .8], 2), .0003)
        self.assertEqual(partial_conjunction([.1], 2), 1)
        self.assertEqual(partial_conjunction([.1, .2, .3], 3), .3)

    def test_null_one_adversarial_guide_probability(self):
        rng = np.random.default_rng(16151)
        p = np.column_stack([np.zeros(50000), rng.uniform(size=(50000, 3))])
        combined = 3 * np.sort(p, axis=1)[:, 1]
        self.assertLessEqual(float((combined < .05).mean()), .055)

    def test_invalid_parameters_and_by(self):
        for p, required in [([np.nan], 1), ([-1], 1), ([1.1], 1), ([.1], 0)]:
            with self.assertRaises(ValueError):
                partial_conjunction(p, required)
        p = [.001, .01, .1]
        self.assertTrue(np.all(adjust_pvalues(p, "by") >= adjust_pvalues(p, "bh")))


class GuideInferenceTests(unittest.TestCase):
    def setUp(self):
        self.table, self.meta, self.truth, _ = simulate(SimulationConfig(genes=150, seed=842))

    def test_reverse_contrast_preserves_p_and_median_effect(self):
        a, ai, _ = call_guide_hits(self.table, self.meta, "treatment", "control")
        b, bi, _ = call_guide_hits(self.table, self.meta, "control", "treatment")
        self.assertEqual(ai["residual_df_per_guide"], 4)
        for left, right in zip(a, b):
            self.assertAlmostEqual(left["log2_effect"], -right["log2_effect"], places=10)
            if left["tested"]:
                self.assertAlmostEqual(left["pvalue"], right["pvalue"], places=10)

    def test_one_strong_guide_cannot_supply_two_guide_evidence(self):
        table, meta, _, _ = simulate(SimulationConfig(genes=300, active_fraction=0, seed=55132))
        x = table.counts.copy()
        x[0, 3:] *= 100000
        # Use valid declared controls so the artifact cannot alter normalization.
        rows, info, guides = call_guide_hits(CountTable(table.ids, table.genes, table.samples, x), meta,
                                            "treatment", "control", GuideConfig(normalization="control"))
        first = rows[0]
        self.assertEqual(first["required_support"], 2)
        self.assertIn("g00001_sg1", first["outlier_guides"])
        self.assertFalse(first["hit"])
        self.assertEqual(info["removed_outlier_guides"], 0)
        self.assertTrue(all(not r["removed_from_test"] for r in guides))

    def test_low_information_guides_have_one_pvalue(self):
        x = self.table.counts.copy()
        x[0] = 0
        rows, info, guides = call_guide_hits(CountTable(self.table.ids, self.table.genes, self.table.samples, x), self.meta,
                                            "treatment", "control")
        self.assertFalse(guides[0]["informative"])
        self.assertEqual(guides[0]["p_enrichment"], 1)
        self.assertEqual(guides[0]["p_depletion"], 1)
        self.assertEqual(rows[0]["guides"], 4)
        self.assertEqual(rows[0]["required_support"], 2)
        self.assertEqual(rows[0]["informative_guides"], 3)

    def test_sample_order_invariance(self):
        rows, _, _ = call_guide_hits(self.table, self.meta, "treatment", "control")
        order = [5, 3, 0, 2, 1, 4]
        table = CountTable(self.table.ids, self.table.genes, [self.table.samples[i] for i in order], self.table.counts[:, order])
        reordered, _, _ = call_guide_hits(table, self.meta, "treatment", "control")
        np.testing.assert_allclose([r["pvalue"] for r in rows], [r["pvalue"] for r in reordered], rtol=1e-9)

    def test_support_requirement_fixed_by_annotation(self):
        for k, required in ((2, 2), (3, 2), (4, 2), (6, 3), (8, 4)):
            table, meta, _, _ = simulate(SimulationConfig(genes=30, guides_per_gene=k, seed=31))
            rows, info, _ = call_guide_hits(table, meta, "treatment", "control")
            self.assertTrue(all(r["required_support"] == required for r in rows))
            self.assertEqual(info["residual_df_per_guide"], 4)

    def test_paired_design_does_not_gain_df_from_guides(self):
        table, meta, _, _ = simulate(SimulationConfig(genes=50, paired=True, seed=313))
        _, info, _ = call_guide_hits(table, meta, "treatment", "control")
        self.assertEqual(info["residual_df_per_guide"], 2)
        self.assertFalse(info["guide_independence_assumed_for_combination"])

    def test_configuration_validation(self):
        for config in (GuideConfig(minimum_support=1), GuideConfig(support_fraction=0), GuideConfig(adjustment="none")):
            with self.assertRaises(ValueError):
                call_guide_hits(self.table, self.meta, "treatment", "control", config)


if __name__ == "__main__":
    unittest.main()
