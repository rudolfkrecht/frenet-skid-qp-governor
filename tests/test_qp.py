import unittest
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from governor.qp import Parameters, govern, solve_qp_2d, feasible


class QPTests(unittest.TestCase):
    def test_infeasible_never_claims_optimum(self):
        self.assertIsNone(solve_qp_2d(2,2,0,0,[(1,0,-1),(-1,0,0)]))

    def test_known_edge_and_vertex(self):
        self.assertEqual(solve_qp_2d(2,2,-4,-6,[(1,0,1),(0,1,1)]),(1.,1.))
        self.assertAlmostEqual(solve_qp_2d(2,2,-4,0,[(1,0,1)])[0],1.)

    def test_forward_curvature_torque_and_slew(self):
        p=Parameters()
        previous=(0.,0.)
        for _ in range(30):
            result=govern((.9,2.),previous,previous,p)
            self.assertEqual(result.status,'optimal')
            self.assertTrue(feasible(result.command,result.constraints))
            self.assertLessEqual(abs(result.command[1]),p.kappa_max*result.command[0]+1e-8)
            self.assertLessEqual(abs(result.command[0]-previous[0]),p.dv_dt_max*p.sample_time+1e-8)
            self.assertLessEqual(abs(result.torques[0]),p.eta_tau*p.tau_sum_max+1e-8)
            previous=result.command

    def test_random_qps_against_independent_scipy_optimizer(self):
        rng=np.random.default_rng(296)
        for _ in range(60):
            A=rng.normal(size=(12,2));b=rng.uniform(.1,2.,12)
            H=rng.uniform(.4,8.,2);f=rng.normal(size=2)*4
            rows=[(*a,c) for a,c in zip(A,b)]
            x=np.array(solve_qp_2d(*H,*f,rows))
            fun=lambda z:.5*np.dot(H*z,z)+f@z
            opt=minimize(fun,np.zeros(2),jac=lambda z:H*z+f,
                         constraints={'type':'ineq','fun':lambda z:b-A@z,'jac':lambda z:-A},
                         method='SLSQP',options={'ftol':1e-11,'maxiter':200})
            self.assertTrue(opt.success,opt.message)
            self.assertTrue(feasible(x,rows))
            self.assertAlmostEqual(fun(x),opt.fun,places=7)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): Parameters(b_v=0)
        with self.assertRaises(ValueError): govern((float('nan'),0),(0,0),(0,0))
        p=replace(Parameters(),nonreversing_wheels=True)
        r=govern((.2,.6),(0,0),(0,0),p)
        self.assertEqual(r.status,'optimal')


if __name__=='__main__': unittest.main()
