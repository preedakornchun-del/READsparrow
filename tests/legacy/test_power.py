import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from readsparrow.cli import main
from readsparrow.empirical_inference import call_empirical_hits
from readsparrow.io import CountTable, save_counts, write_tsv
from readsparrow.null_comparison import generate
from readsparrow.power_inference import (PowerConfig, call_power_hits, nuisance_variance,
    combined_df, loo_rows_from_power, winsorized_location)
from readsparrow.simulation import SimulationConfig, simulate


class PowerTests(unittest.TestCase):
    def setUp(self):
        self.table, self.meta, self.truth, _ = simulate(SimulationConfig(genes=100, controls=200, seed=1948))

    def test_shared_location_uncertainty_is_not_divided_by_guides(self):
        self.assertAlmostEqual(nuisance_variance([1]*4, [0]*4, [.1]), .25+.1)
        self.assertAlmostEqual(nuisance_variance([1]*4, [0]*4, [.1], 1), 1+.1)
        self.assertAlmostEqual(nuisance_variance([1]*4, [0,0,1,1], [.1,.2]), .25+.025+.05)

    def test_df_components_not_extra_biological_replication(self):
        self.assertAlmostEqual(float(combined_df(1.,0.,4.,100.)),4.)
        self.assertAlmostEqual(float(combined_df(0.,1.,4.,100.)),100.)
        a, info, _ = call_power_hits(self.table, self.meta, "treatment", "control")
        self.assertEqual(info["residual_df_per_gene"],4)
        self.assertTrue(info["technical_background_independence_assumed"])

    def test_winsorization_keeps_coherent_signal_and_bounds_outlier(self):
        x = np.array([0.,0.,0.,1000.])
        self.assertEqual(winsorized_location(x,np.ones(4)),.625)
        self.assertEqual(winsorized_location(x+10,np.ones(4)),10.625)
        self.assertEqual(winsorized_location([10]*4,np.ones(4)),10.)

    def test_context_export_does_not_change_legacy_outputs(self):
        before = call_empirical_hits(self.table,self.meta,"treatment","control")
        ctx = {}
        after = call_empirical_hits(self.table,self.meta,"treatment","control",_context_sink=ctx)
        self.assertEqual(before,after)
        self.assertEqual(len(ctx["train"]),100)

    def test_loo_derived_decisions_equal_standalone(self):
        a, _, _ = call_power_hits(self.table,self.meta,"treatment","control",PowerConfig(aggregation="bounded"))
        b, _, _ = call_power_hits(self.table,self.meta,"treatment","control",PowerConfig(aggregation="loo"))
        self.assertEqual(loo_rows_from_power(a), b)

    def test_all_annotated_guides_remain_in_family(self):
        x = self.table.counts.copy()
        x[0] = 0
        table = CountTable(self.table.ids,self.table.genes,self.table.samples,x)
        rows, _, _ = call_power_hits(table,self.meta,"treatment","control")
        self.assertFalse(rows[0]["tested"])
        self.assertEqual(rows[0]["guides"],4)
        self.assertIsNone(rows[0]["pvalue"])

    def test_sample_order_invariance(self):
        a,_,_ = call_power_hits(self.table,self.meta,"treatment","control")
        order=[5,3,1,4,0,2]
        table=CountTable(self.table.ids,self.table.genes,[self.table.samples[i] for i in order],self.table.counts[:,order])
        b,_,_ = call_power_hits(table,self.meta,"treatment","control")
        np.testing.assert_allclose([r["pvalue"] for r in a if r["tested"]],[r["pvalue"] for r in b if r["tested"]],rtol=1e-8,atol=1e-12)

    def test_single_outlier_loo_is_not_sufficient_evidence(self):
        table,meta,_,_=simulate(SimulationConfig(genes=100,controls=200,active_fraction=0,seed=971))
        x=table.counts.copy()
        x[0,3:]*=100000
        table=CountTable(table.ids,table.genes,table.samples,x)
        rows,_,_=call_power_hits(table,meta,"treatment","control",PowerConfig(aggregation="loo",normalization="control"))
        self.assertFalse(rows[0]["hit"])

    def test_moderate_effect_development_regression(self):
        table,meta,truth=generate(SimulationConfig(genes=1000,controls=1000,seed=78420000),drift="gaussian")
        before,_,_=call_empirical_hits(table,meta,"treatment","control")
        after,_,_=call_power_hits(table,meta,"treatment","control",PowerConfig(aggregation="bounded"))
        active={r["gene"] for r in truth if r["active"]}
        self.assertEqual(sum(r["hit"] for r in before),0)
        self.assertGreater(sum(r["hit"] and r["gene"] in active for r in after),10)
        # Engineering regression only; this one development screen is not FDR validation.

    def test_invalid_config(self):
        for cfg in (PowerConfig(background_correlation=-1),PowerConfig(clipping_threshold=0),PowerConfig(aggregation="magic"),PowerConfig(null_model="empirical")):
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_both_cli_methods_without_external_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            save_counts(root/"counts.tsv",self.table)
            write_tsv(root/"samples.tsv",list(self.meta[0]),self.meta)
            for method in ("power-bounded","power-loo"):
                with patch("subprocess.run",side_effect=AssertionError("No external engine")),contextlib.redirect_stdout(io.StringIO()):
                    result=main(["test","--counts",str(root/"counts.tsv"),"--samples",str(root/"samples.tsv"),"--treatment","treatment","--control","control","--method",method,"--experimental","--output",str(root/method)])
                self.assertEqual(result,0)
                self.assertTrue((root/method/"gene_subset_audit.tsv").exists())


if __name__ == "__main__":
    unittest.main()
