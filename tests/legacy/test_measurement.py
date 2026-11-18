import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import multivariate_normal,multivariate_t
from readsparrow.cli import main,parser
from readsparrow.empirical_inference import call_empirical_hits
from readsparrow.io import CountTable,save_counts,write_tsv
from readsparrow.shape_comparison import cases,generate
from readsparrow.shape_inference import call_shape_hits
from readsparrow.measurement_inference import (MeasurementConfig,call_measurement_hits,
    psd,correlation,reference_residuals,measurement_covariances,nuisance_draws,
    predictive_logpdf,component_densities,gaussian_components,measurement_mixture,counting_noise_floor)

class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=cases()['matched_ineffective_rho0.5']
        cls.cfg=replace(cls.case['config'],genes=200,controls=400,seed=1830000001)
        cls.table,cls.meta,cls.truth,cls.blocks=generate(cls.cfg,cls.case)

    def test_psd_and_unit_correlation(self):
        a=psd([[1.,2.],[2.,1.]])
        self.assertGreaterEqual(np.linalg.eigvalsh(a).min(),0.)
        np.testing.assert_allclose(np.diag(correlation(a)),1.)

    def test_reference_retains_cross_covariance_exactly(self):
        rng=np.random.default_rng(83);r=rng.normal(size=(12,6))
        assignment=np.zeros(12,dtype=int);training=np.arange(4)
        adjusted,ref=reference_residuals(r,assignment,training)
        a=r[4:8];b=ref[4:8]
        expected=a@a.T+b@b.T-a@b.T-b@a.T
        np.testing.assert_allclose(adjusted[4:8]@adjusted[4:8].T,expected,atol=1e-12)
        self.assertGreater(np.max(np.abs(expected-(a@a.T+b@b.T))),.01)

    def test_shared_measurement_can_cancel(self):
        r=np.repeat(np.array([[1.,-1.,2.,-2.]]),8,axis=0)
        out,_=reference_residuals(r,np.zeros(8,dtype=int),np.arange(4))
        np.testing.assert_allclose(out,0.)

    def test_independent_read_noise_cannot_cancel(self):
        design=np.column_stack([np.ones(6),np.array([0,0,0,1,1,1])])
        covariance=np.ones((1,3,3))*.2
        out=counting_noise_floor(covariance,[np.arange(3)],np.ones((3,6))*100,design,.5)
        self.assertGreater(np.linalg.eigvalsh(out[0]).min(),.01)

    def test_center_measurement_uncertainty_not_double_bootstrapped(self):
        x=np.random.default_rng(83).normal(size=(100,3))
        draws=nuisance_draws(x,np.repeat(np.eye(3)[None]*.5,100,axis=0),12,73)
        for mean,_,center_cov in draws:
            np.testing.assert_allclose(mean,0.)
            self.assertTrue(np.linalg.eigvalsh(center_cov).min()>-1e-10)

    def test_student_variance_parameterization(self):
        x=np.array([[.4,1.1],[-1.,2.]])
        cov=np.array([[2.,.4],[.4,1.]])
        found=predictive_logpdf(x,np.zeros_like(x),np.repeat(cov[None],2,axis=0),np.repeat(12.,2))
        expected=multivariate_t.logpdf(x,shape=cov*10/12,df=12)
        np.testing.assert_allclose(found,expected)

    def test_gaussian_component_clean_and_prior_normalization(self):
        cov=np.array([[[1.,.3],[.3,2.]]]);x=np.array([[.3,-.7]])
        result=gaussian_components(x,np.zeros_like(x),cov)
        self.assertAlmostEqual(result[0,0],multivariate_normal.logpdf(x[0],cov=cov[0]))
        # Independent brute-force fixed multi prior for two guides.
        density=np.mean([multivariate_normal.pdf(x[0],mean=s*e*np.ones(2),cov=cov[0])
            for e in (.75,1.5,2.5,4.) for s in (-1.,1.)])
        self.assertAlmostEqual(np.exp(result[2,0]),density)

    def test_quadrature_weights_and_measurement_variance(self):
        scales,weights=measurement_mixture(40.,7)
        self.assertAlmostEqual(np.exp(weights).sum(),1.)
        self.assertAlmostEqual(np.sum(scales*np.exp(weights)),1.,places=5)

    def test_gaussian_components_sign_and_permutation(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(4,3));c=np.eye(3)+.4*np.ones((3,3))
        cov=np.repeat(c[None],4,axis=0)
        a=gaussian_components(x,np.zeros_like(x),cov)
        np.testing.assert_allclose(a,gaussian_components(-x,np.zeros_like(x),cov))
        np.testing.assert_allclose(a,gaussian_components(x[:,::-1],np.zeros_like(x),cov[:,::-1,::-1]))

    def test_block_bootstrap_reproducible_and_noise_subtraction(self):
        rng=np.random.default_rng(73);x=rng.normal(size=(1000,3))
        noise=np.repeat(np.eye(3)[None]*.5,1000,axis=0)
        a=nuisance_draws(x,noise,12,8383);b=nuisance_draws(x,noise,12,8383)
        for aa,bb in zip(a,b):
            for aaa,bbb in zip(aa,bb):np.testing.assert_allclose(aaa,bbb)
        np.testing.assert_allclose(np.diag(a[0][1]),.5,atol=.12)

    def test_separate_control_target_measurement(self):
        _,info,_=call_measurement_hits(self.table,self.meta,'treatment','control',
            MeasurementConfig(correlation_mode='matched-blocks'),control_blocks=self.blocks)
        model=info['measurement_model'];self.assertIsNotNone(model)
        target=np.array(model['target_correlation']['4']);control=np.array(model['control_correlation'])
        self.assertGreater(target[np.triu_indices(4,1)].mean(),control[np.triu_indices(4,1)].mean()+.1)
        self.assertEqual(model['biological_residual_df'],4)

    def test_missing_blocks_refused(self):
        with self.assertRaises(ValueError):
            call_measurement_hits(self.table,self.meta,'treatment','control',MeasurementConfig(correlation_mode='matched-blocks'))

    def test_no_blocks_legacy_guard_exact(self):
        a,_,_=call_shape_hits(self.table,self.meta,'treatment','control')
        b,info,_=call_measurement_hits(self.table,self.meta,'treatment','control')
        self.assertEqual([(r['qvalue'],r['hit']) for r in a],[(r['qvalue'],r['hit']) for r in b])
        self.assertIsNone(info['measurement_model'])
        self.assertEqual(info['confidence_interval'],'Not supplied: descriptive median effect only')

    def test_prepared_context_exact(self):
        context={};rows,info,guides=call_empirical_hits(self.table,self.meta,'treatment','control',_context_sink=context)
        cfg=MeasurementConfig(correlation_mode='matched-blocks')
        a,_,_=call_measurement_hits(self.table,self.meta,'treatment','control',cfg,control_blocks=self.blocks)
        b,_,_=call_measurement_hits(self.table,self.meta,'treatment','control',cfg,control_blocks=self.blocks,_prepared=(rows,info,guides,context))
        self.assertEqual([r['qvalue'] for r in a],[r['qvalue'] for r in b])

    def test_row_order_invariance(self):
        cfg=MeasurementConfig(correlation_mode='matched-blocks')
        a,_,_=call_measurement_hits(self.table,self.meta,'treatment','control',cfg,control_blocks=self.blocks)
        perm=np.random.default_rng(983).permutation(len(self.table.ids))
        t=CountTable([self.table.ids[i] for i in perm],[self.table.genes[i] for i in perm],self.table.samples,self.table.counts[perm])
        b,_,_=call_measurement_hits(t,self.meta,'treatment','control',cfg,control_blocks=self.blocks)
        np.testing.assert_allclose([r['qvalue'] for r in a if r['tested']],[r['qvalue'] for r in b if r['tested']],atol=1e-6)

    def test_cli_standalone_and_default_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);save_counts(root/'counts.tsv',self.table)
            write_tsv(root/'samples.tsv',list(self.meta[0]),self.meta)
            write_tsv(root/'blocks.tsv',['sgRNA','block'],[dict(sgRNA=g,block=b) for g,b in self.blocks.items()])
            with patch('subprocess.run',side_effect=AssertionError('No external engine')),contextlib.redirect_stdout(io.StringIO()):
                code=main(['test','--counts',str(root/'counts.tsv'),'--samples',str(root/'samples.tsv'),
                    '--treatment','treatment','--control','control','--method','measurement-adaptive',
                    '--control-blocks',str(root/'blocks.tsv'),'--experimental','--output',str(root/'out')])
            self.assertEqual(code,0);report=json.loads((root/'out/analysis.json').read_text())
            self.assertIn('NOT BH',report['fdr_family'])
            self.assertIsNotNone(report['measurement_model'])
        self.assertEqual(parser().parse_args(['test','--counts','a','--samples','b','--treatment','t','--control','c','--output','o']).method,'background')

if __name__=='__main__':unittest.main()
