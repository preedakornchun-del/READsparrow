"""Boundary/Jacobian/effect-scale regression tests, independent of gene truth."""
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from scipy.optimize import check_grad,OptimizeResult
from readsparrow.stable_null import SharedNullLikelihood
from readsparrow.stable_inference import StableConfig,call_stable_hits
from readsparrow.joint_inference import call_joint_hits
from readsparrow.measurement_comparison import cases,generate
from readsparrow.cli import parser

class StableTests(unittest.TestCase):
    def engine(self,learn=True):
        r=np.random.default_rng(2510000001)
        return SharedNullLikelihood([(r.normal(size=(50,4)),np.tile(np.eye(4)*.2,(50,1,1)))],
            r.normal(size=(50,4)),np.tile(np.eye(4)*.1,(50,1,1)),learn_scale=learn)

    def test_effect_scale_gradient(self):
        e=self.engine();z=np.array([.7,.4,.05,.1,.1,.35,.3])
        self.assertLess(check_grad(lambda x:e.evaluate(x)[0],lambda x:e.evaluate(x)[1],z),1e-3)

    def test_transformed_gradient(self):
        e=self.engine();u=np.array([-4.,-5.,0.,-2.,-3.,.4,-.2])
        self.assertLess(check_grad(lambda x:e.transformed(x,False)[0],lambda x:e.transformed(x,False)[1],u),1e-3)

    def test_measurement_student_mixture_gradient(self):
        e=self.engine();e.measurement_df=(12.,18.);e.measurement_quadrature=3
        z=np.array([.7,.4,.05,.1,.1,.35,.3])
        self.assertLess(check_grad(lambda x:e.evaluate(x)[0],lambda x:e.evaluate(x)[1],z),1e-3)

    def test_student_quadrature_constant_density_normalized(self):
        e=self.engine();e.measurement_df=(12.,18.);e.measurement_quadrature=3
        def constant(x,m,p,components=True,derivative=False):
            value=np.full((len(x),3),-2.) if components else np.full(len(x),-2.)
            return (value,np.zeros(value.shape+(5 if components else 3,))) if derivative else value
        with patch.object(e,'gaussian_density',side_effect=constant):
            found=e.density(e.c,e.cm,[.1,.2,0,.1,.1,.5,0])
        np.testing.assert_allclose(found,-2.,atol=1e-15)

    def test_jacobian_and_support(self):
        for d in (4,6,7):
            u=np.linspace(-5,4,d);theta,lj,jac,_=SharedNullLikelihood.transform(u)
            self.assertAlmostEqual(lj,np.log(jac).sum(),places=12)
            for j in range(d):
                hi=u.copy();lo=u.copy();hi[j]+=1e-5;lo[j]-=1e-5
                numerical=(SharedNullLikelihood.transform(hi)[0][j]-SharedNullLikelihood.transform(lo)[0][j])/2e-5
                self.assertAlmostEqual(numerical,jac[j],places=7)
            self.assertTrue(np.isfinite(theta).all())

    def test_fixed_likelihood_identical_to_previous_interior(self):
        from readsparrow.joint_null import SharedNullLikelihood as Previous
        e=self.engine(False);old=Previous(e.groups,e.c,e.cm);z=[.7,.4,.05,.1,.1,.35]
        self.assertEqual(e.evaluate(z)[0],old.evaluate(z)[0])
        np.testing.assert_equal(e.evaluate(z)[1],old.evaluate(z)[1])

    def test_zero_mixture_has_exact_zero_posterior(self):
        e=self.engine();_,local=e.evaluate([.7,.4,.05,0.,0.,.5,0.],details=True)
        np.testing.assert_equal(local,np.ones(len(local)))

    def test_uniform_evidence_has_correct_jacobian_and_prior_mass(self):
        e=self.engine(False)
        def flat(z,null=False,gradient=True,local=False,details=False):
            if details:return 0.,np.ones(50)
            return (0.,np.zeros(len(z))) if gradient else 0.
        with patch.object(e,'evaluate',side_effect=flat):
            _,z0,_=e.integrate(OptimizeResult(x=np.array([50.,50.,0.,.5]),fun=0.),True,4096,2510000002)
            _,z1,_=e.integrate(OptimizeResult(x=np.array([50.,50.,0.,.5,.25,.5]),fun=0.),False,4096,2510000003)
        expected=np.log(100*100*16)
        self.assertLess(abs(z0-expected),.1);self.assertLess(abs(z1-expected),.15)

    def test_reproducibility_and_finite_rho_one_covariance(self):
        e=self.engine(False);e.c[:]=e.c[:,:1];e.cm[:]=np.eye(4)*.01
        first=e.decide(draws=32,seed=2510000004);second=e.decide(draws=32,seed=2510000004)
        np.testing.assert_equal(first[0],second[0]);np.testing.assert_equal(first[1],second[1])
        self.assertTrue(np.isfinite(first[1]).all())
        self.assertEqual(first[2]['full_integration']['proposal_support_probability'],1.)

    def test_no_blocks_fallback_exact_and_default_unchanged(self):
        case=cases()['nb_mixture_n3'];cfg=replace(case['config'],genes=100,seed=2510000005)
        table,meta,_,_=generate(cfg,case)
        old=call_joint_hits(table,meta,'treatment','control')[0]
        new=call_stable_hits(table,meta,'treatment','control')[0]
        self.assertEqual(old,new)
        self.assertEqual(parser().parse_args(['test','--counts','a','--samples','b','--treatment','t','--control','c','--output','o']).method,'background')

    def test_draw_limits_and_support_override(self):
        for cfg in (StableConfig(shared_draws=7),StableConfig(shared_draws=4097),StableConfig(minimum_support=3)):
            with self.assertRaises(ValueError):cfg.validate()

    def test_cli_stable_is_opt_in(self):
        args=parser().parse_args(['test','--counts','a','--samples','b','--treatment','t','--control','c','--method','stable-adaptive','--experimental','--output','o'])
        self.assertEqual(args.method,'stable-adaptive');self.assertTrue(args.experimental)
        self.assertEqual(StableConfig().stable_measurement_quadrature,3)

if __name__=='__main__':unittest.main()
