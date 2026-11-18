import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from readsparrow.cli import main, parser
from readsparrow.correlated_comparison import cases, generate
from readsparrow.correlated_inference import CorrelatedConfig, adaptive_directional, call_correlated_hits, fit_correlation, projected_nuisance
from readsparrow.empirical_inference import EmpiricalConfig, call_empirical_hits
from readsparrow.io import CountTable, save_counts, write_tsv
from readsparrow.simulation import SimulationConfig, simulate


class CorrelatedTests(unittest.TestCase):
    def setUp(self):
        self.table,self.meta,_,_=simulate(SimulationConfig(genes=100,controls=200,seed=4621))

    def test_default_is_not_independence(self):
        rows,info,_=call_correlated_hits(self.table,self.meta,"treatment","control")
        self.assertEqual(info["correlation_model"]["upper"],1.)
        self.assertEqual(info["residual_df_per_gene"],4)
        self.assertTrue(all(r["required_support"]==2 for r in rows))

    def test_weighted_union_is_corrected(self):
        self.assertAlmostEqual(adaptive_directional(.01,.4,.8),.0125)
        self.assertAlmostEqual(adaptive_directional(.4,.01,.8),.05)
        self.assertEqual(adaptive_directional(1.,1.,.8),1.)

    def test_invalid_config(self):
        for cfg in (CorrelatedConfig(correlation_upper=-1),CorrelatedConfig(correlation_upper=1.1),
            CorrelatedConfig(correlation_mode="magic"),CorrelatedConfig(coherent_weight=0),CorrelatedConfig(min_control_blocks=3),
            CorrelatedConfig(minimum_support=3),CorrelatedConfig(null_model="empirical")):
            with self.assertRaises(ValueError):cfg.validate()

    def test_no_silent_matched_block_fallback(self):
        with self.assertRaisesRegex(ValueError,"requires genuine"):
            call_correlated_hits(self.table,self.meta,"treatment","control",CorrelatedConfig(correlation_mode="matched-blocks"))

    def test_no_unused_blocks_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,"mode is not"):
            call_correlated_hits(self.table,self.meta,"treatment","control",control_blocks={"NTC_0001":"a"})

    def test_target_cannot_be_null_control_block(self):
        with self.assertRaisesRegex(ValueError,"known informative"):
            call_correlated_hits(self.table,self.meta,"treatment","control",CorrelatedConfig(correlation_mode="matched-blocks"),control_blocks={self.table.ids[0]:"fake"})

    def test_unbalanced_blocks_rejected(self):
        blocks={f"NTC_{i+1:04d}":str(i//4) for i in range(199)}
        with self.assertRaisesRegex(ValueError,"balanced"):
            call_correlated_hits(self.table,self.meta,"treatment","control",CorrelatedConfig(correlation_mode="matched-blocks"),control_blocks=blocks)

    def test_sparse_information_filter_not_effect_filter(self):
        table,meta,_,_=simulate(SimulationConfig(genes=100,controls=200,seed=491,ineffective_fraction=.75))
        rows,_,guides=call_correlated_hits(table,meta,"treatment","control")
        self.assertGreater(sum(r["tested"] for r in rows),90)
        self.assertFalse(any(r["removed_from_test"] for r in guides))
        x=table.counts.copy();x[0]=0
        filtered,_,_=call_correlated_hits(CountTable(table.ids,table.genes,table.samples,x),meta,"treatment","control")
        self.assertFalse(filtered[0]["tested"])
        self.assertIsNone(filtered[0]["pvalue"])

    def test_single_huge_guide_does_not_suffice(self):
        table,meta,_,_=simulate(SimulationConfig(genes=100,controls=200,active_fraction=0,seed=613))
        x=table.counts.copy();x[0,3:]*=100000
        rows,_,_=call_correlated_hits(CountTable(table.ids,table.genes,table.samples,x),meta,"treatment","control",CorrelatedConfig(normalization="control"))
        self.assertFalse(rows[0]["hit"])

    def test_sample_order_invariance(self):
        rows,_,_=call_correlated_hits(self.table,self.meta,"treatment","control")
        order=[5,3,1,4,0,2]
        perm=CountTable(self.table.ids,self.table.genes,[self.table.samples[i] for i in order],self.table.counts[:,order])
        after,_,_=call_correlated_hits(perm,self.meta,"treatment","control")
        np.testing.assert_allclose([r["pvalue"] for r in rows if r["tested"]],[r["pvalue"] for r in after if r["tested"]],rtol=1e-8,atol=1e-12)

    def test_raw_normal_icc_upper_bounds_known_rho(self):
        rng=np.random.default_rng(567);n,k,rho=500,4,.5
        ids=[str(i) for i in range(n*k)]
        z=(np.sqrt(rho)*rng.normal(size=(n,1))+np.sqrt(1-rho)*rng.normal(size=(n,k))).ravel()
        context=dict(z=z,train=np.arange(n*k//2),calibration=np.arange(n*k//2,n*k),bins=[dict(extra_background_variance=1.)],
            assignment=np.zeros(n*k,dtype=int),prediction_se=np.ones(n*k))
        info=fit_correlation(context,CountTable(ids,["NA"]*len(ids),["a"],np.ones((len(ids),1),dtype=int)),
            {gid:str(i//k) for i,gid in enumerate(ids)},CorrelatedConfig(correlation_mode="matched-blocks"))
        self.assertGreaterEqual(info["upper"],rho)
        self.assertLess(info["upper"],.8)
        self.assertEqual(info["lower"],0.)

    def test_matched_controls_generate_shared_not_random_nuisance(self):
        case=cases()["matched_null_rho0.5"]
        cfg=replace(case["config"],genes=100,seed=943,controls=200)
        table,meta,_,blocks=generate(cfg,case)
        self.assertEqual(len(set(blocks.values())),50)
        rows,info,_=call_correlated_hits(table,meta,"treatment","control",CorrelatedConfig(correlation_mode="matched-blocks"),control_blocks=blocks)
        self.assertEqual(info["correlation_model"]["supplied_blocks"],50)
        self.assertGreaterEqual(info["correlation_model"]["blocks"],40)
        self.assertGreaterEqual(info["correlation_model"]["upper"],.5)
        self.assertEqual(info["residual_df_per_gene"],4)

    def test_previous_model_unchanged(self):
        a=call_empirical_hits(self.table,self.meta,"treatment","control",EmpiricalConfig())
        call_correlated_hits(self.table,self.meta,"treatment","control")
        b=call_empirical_hits(self.table,self.meta,"treatment","control",EmpiricalConfig())
        self.assertEqual(a,b)

    def test_projected_covariance_not_independent_pseudo_blocks(self):
        pop=dict(mean=np.zeros(4),covariance=np.eye(4),blocks=100,size=4)
        var,center=projected_nuisance(np.array([0,1,2]),np.array([1.]),np.zeros(4,dtype=int),np.array([.1]),pop)
        self.assertAlmostEqual(var,(1+1/100)/3+.1)
        self.assertEqual(center,0.)
        pop["covariance"]=np.ones((4,4))
        correlated,_=projected_nuisance(np.array([0,1,2]),np.array([1.]),np.zeros(4,dtype=int),np.array([.1]),pop)
        self.assertAlmostEqual(correlated,1.01+.1)

    def test_matched_guide_row_order_invariance(self):
        case=cases()["matched_mixture_rho0"];cfg=replace(case["config"],genes=100,controls=200,seed=943)
        table,meta,_,blocks=generate(cfg,case);config=CorrelatedConfig(correlation_mode="matched-blocks")
        before,_,_=call_correlated_hits(table,meta,"treatment","control",config,control_blocks=blocks)
        order=np.random.default_rng(736).permutation(len(table.ids))
        perm=CountTable([table.ids[i] for i in order],[table.genes[i] for i in order],table.samples,table.counts[order])
        after,_,_=call_correlated_hits(perm,meta,"treatment","control",config,control_blocks=blocks)
        np.testing.assert_allclose([r["pvalue"] for r in before if r["tested"]],[r["pvalue"] for r in after if r["tested"]],rtol=1e-8,atol=1e-12)

    def test_cli_matched_blocks_records_input_and_model(self):
        case=cases()["matched_null_rho0.5"];cfg=replace(case["config"],genes=100,controls=200,seed=943)
        table,meta,_,blocks=generate(cfg,case)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);save_counts(root/"counts.tsv",table);write_tsv(root/"samples.tsv",list(meta[0]),meta)
            write_tsv(root/"blocks.tsv",["sgRNA","block"],[dict(sgRNA=g,block=b) for g,b in blocks.items()])
            with patch("subprocess.run",side_effect=AssertionError("No external engine")),contextlib.redirect_stdout(io.StringIO()):
                code=main(["test","--counts",str(root/"counts.tsv"),"--samples",str(root/"samples.tsv"),"--treatment","treatment","--control","control",
                    "--method","corr-adaptive","--control-blocks",str(root/"blocks.tsv"),"--experimental","--output",str(root/"out")])
            self.assertEqual(code,0)
            info=json.loads((root/"out/analysis.json").read_text())
            self.assertEqual(info["correlation_model"]["main_variance_model"],"matched-projection-predictive")
            cov=np.array(info["correlation_model"]["projection_population_fit"]["nuisance_covariance"])
            self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-12)
            prov=json.loads((root/"out/provenance.json").read_text())
            self.assertIn(str((root/"blocks.tsv").resolve()),[r["path"] for r in prov["inputs"]])

    def test_cli_new_method_has_no_external_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);save_counts(root/"counts.tsv",self.table);write_tsv(root/"samples.tsv",list(self.meta[0]),self.meta)
            with patch("subprocess.run",side_effect=AssertionError("No external engine")),contextlib.redirect_stdout(io.StringIO()):
                code=main(["test","--counts",str(root/"counts.tsv"),"--samples",str(root/"samples.tsv"),"--treatment","treatment","--control","control",
                    "--method","corr-adaptive","--experimental","--output",str(root/"out")])
            self.assertEqual(code,0)
            self.assertTrue((root/"out/gene_subset_audit.tsv").exists())
            info=json.loads((root/"out/analysis.json").read_text())
            self.assertEqual(info["correlation_model"]["upper"],1.)
            self.assertEqual(parser().parse_args(["test","--counts","a","--samples","b","--treatment","t","--control","c","--output","o"]).method,"background")


if __name__=="__main__":unittest.main()
