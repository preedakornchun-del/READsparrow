import contextlib,io,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from scipy.optimize import check_grad
from scipy.stats import multivariate_normal,norm
from readsparrow.joint_null import SharedNullLikelihood,masks_and_priors
from readsparrow.joint_inference import JointConfig,call_joint_hits
from readsparrow.measurement_comparison import cases,generate
from readsparrow.shape_inference import call_shape_hits
from readsparrow.empirical_inference import call_empirical_hits
from readsparrow.cli import main,parser
from readsparrow.io import CountTable,save_counts,write_tsv

class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=cases()['matched_ineffective_rho0.5']
        cls.cfg=replace(cls.case['config'],genes=200,controls=400,seed=2280000001)
        cls.table,cls.meta,cls.truth,cls.blocks=generate(cls.cfg,cls.case)

    def engine(self):
        rng=np.random.default_rng(73);x=rng.normal(size=(50,3));c=rng.normal(size=(60,4))
        return SharedNullLikelihood([(x,np.repeat(np.eye(3)[None]*.2,50,axis=0))],c,np.repeat(np.eye(4)[None]*.1,60,axis=0))

    def test_class_priors_sum_to_one(self):
        for m in (2,3,4,6,8):
            _,_,classes,prior,_=masks_and_priors(m)
            for j in range(3):self.assertAlmostEqual(np.exp(prior[classes==j]).sum(),1.)

    def test_clean_density_exact(self):
        e=self.engine();x=np.array([[.2,-.3,.7]]);m=np.array([np.eye(3)*.2]);t=[.4,1.2,.1,.1,.1]
        result=e.density(x,m,t)
        cov=m[0]+.4*(np.eye(3)-np.ones((3,3))/4)+1.2*np.ones((3,3))/4+np.eye(3)*1e-8
        self.assertAlmostEqual(result[0,0],multivariate_normal.logpdf(x[0],mean=np.repeat(.1,3),cov=cov))

    def test_coherent_density_matches_direct_integral(self):
        e=self.engine();x=np.array([[.4,-.7]]);m=np.array([np.eye(2)*.2]);e.spec[2]=masks_and_priors(2)
        t=[.4,1.2,.1,.1,.1];found=np.exp(e.density(x,m,t)[0,2]);mask=np.ones(2)
        cov=m[0]+.4*(np.eye(2)-np.ones((2,2))/4)+1.2*np.ones((2,2))/4+np.eye(2)*1e-8
        expected=quad(lambda delta:delta**2*norm.pdf(delta)*multivariate_normal.pdf(x[0],mean=.1+delta*mask,cov=cov),-12,12,epsabs=1e-12)[0]
        self.assertAlmostEqual(found,expected,places=11)

    def test_analytic_gradient(self):
        e=self.engine();t=np.array([.7,.4,.05,.1,.1])
        err=check_grad(lambda z:e.evaluate(z)[0],lambda z:e.evaluate(z)[1],t)
        self.assertLess(err,1e-3)

    def test_null_gradient(self):
        e=self.engine();t=np.array([.7,.4,.05,.1])
        err=check_grad(lambda z:e.evaluate(z,True)[0],lambda z:e.evaluate(z,True)[1],t)
        self.assertLess(err,1e-3)

    def test_adaptive_support_gradient(self):
        e=self.engine();t=np.array([.7,.4,.05,.1,.1,.35])
        self.assertLess(check_grad(lambda z:e.evaluate(z)[0],lambda z:e.evaluate(z)[1],t),1e-3)

    def test_adaptive_support_endpoints_finite(self):
        e=self.engine()
        for eta in (0.,1.):
            objective,gradient=e.evaluate([.7,.4,.05,.1,.1,eta])
            self.assertTrue(np.isfinite(objective));self.assertTrue(np.isfinite(gradient).all())

    def test_density_sign_and_position_exchangeability(self):
        e=self.engine();x,m=e.groups[0];t=[.7,.4,0.,.1,.1]
        np.testing.assert_allclose(e.density(x,m,t),e.density(-x,m,t),atol=1e-12)
        np.testing.assert_allclose(e.density(x,m,t),e.density(x[:,::-1],m[:,::-1,::-1],t),atol=1e-12)

    def test_joint_likelihood_is_product_before_integration(self):
        e=self.engine();t=[.7,.4,0.,.1,.1];x,m=e.groups[0]
        clean=e.density(e.c,e.cm,t,False).sum()
        logs=e.density(x,m,t);weights=np.array([.9*.9,.9*.1,.1])
        from scipy.special import logsumexp
        expected=clean+logsumexp(logs+np.log(weights),axis=1).sum()-(t[0]+t[1])/16
        self.assertAlmostEqual(-e.evaluate(t)[0],expected)

    def test_q_arithmetic_reproducible(self):
        e=self.engine();a,q,info=e.decide(draws=16);b,q2,_=e.decide(draws=16)
        np.testing.assert_equal(a,b);np.testing.assert_equal(q,q2)
        order=np.argsort(a,kind='stable');np.testing.assert_allclose(q[order],np.cumsum(a[order])/np.arange(1,len(a)+1))
        self.assertIn('NOT BH',info['family']);self.assertIn('SHARED',info['target_use'])

    def test_no_control_blocks_fallback_all_fields_exact(self):
        old,_,_=call_shape_hits(self.table,self.meta,'treatment','control')
        new,info,_=call_joint_hits(self.table,self.meta,'treatment','control')
        self.assertEqual(old,new);self.assertIsNone(info['joint_null_model'])

    def test_non_gaussian_controls_fallback(self):
        model=dict(family='sinh-arcsinh')
        with patch('readsparrow.joint_inference.fit_shape',return_value=model):
            rows,info,_=call_joint_hits(self.table,self.meta,'treatment','control',JointConfig(correlation_mode='matched-blocks'),control_blocks=self.blocks)
        self.assertIsNone(info['joint_null_model'])

    def test_matched_missing_blocks_refused(self):
        with self.assertRaises(ValueError):call_joint_hits(self.table,self.meta,'treatment','control',JointConfig(correlation_mode='matched-blocks'))

    def test_support_override_refused(self):
        for cfg in (JointConfig(minimum_support=3),JointConfig(support_fraction=.75),JointConfig(shared_draws=2)):
            with self.assertRaises(ValueError):cfg.validate()

    def test_matched_model_no_pvalues(self):
        rows,info,_=call_joint_hits(self.table,self.meta,'treatment','control',JointConfig(correlation_mode='matched-blocks'),control_blocks=self.blocks)
        self.assertIn('shared-null',info['inference_branch']);self.assertTrue(all(r['pvalue'] is None for r in rows))
        self.assertEqual(info['joint_null_model']['control_blocks'],100)
        self.assertEqual(info['measurement_model']['biological_residual_df'],4)

    def test_prepared_context_same(self):
        context={};a,b,c=call_empirical_hits(self.table,self.meta,'treatment','control',_context_sink=context)
        cfg=JointConfig(correlation_mode='matched-blocks')
        first=call_joint_hits(self.table,self.meta,'treatment','control',cfg,control_blocks=self.blocks)
        second=call_joint_hits(self.table,self.meta,'treatment','control',cfg,control_blocks=self.blocks,_prepared=(a,b,c,context))
        self.assertEqual(first[0],second[0])

    def test_cli_standalone_and_default_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);save_counts(root/'counts.tsv',self.table)
            write_tsv(root/'samples.tsv',list(self.meta[0]),self.meta)
            write_tsv(root/'blocks.tsv',['sgRNA','block'],[dict(sgRNA=g,block=b) for g,b in self.blocks.items()])
            with patch('subprocess.run',side_effect=AssertionError('No external engine')),contextlib.redirect_stdout(io.StringIO()):
                code=main(['test','--counts',str(root/'counts.tsv'),'--samples',str(root/'samples.tsv'),'--treatment','treatment','--control','control','--method','joint-adaptive','--control-blocks',str(root/'blocks.tsv'),'--experimental','--output',str(root/'output')])
            self.assertEqual(code,0)
            info=json.loads((root/'output/analysis.json').read_text());self.assertIn('shared-null',info['inference_branch'])
            self.assertIsNone(info['joint_null_model'].get('truth'))
            self.assertEqual(parser().parse_args(['test','--counts','a','--samples','b','--treatment','t','--control','c','--output','o']).method,'background')

if __name__=='__main__':unittest.main()
