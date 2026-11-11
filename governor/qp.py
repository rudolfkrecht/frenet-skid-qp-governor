from dataclasses import dataclass
from itertools import combinations
import math


@dataclass(frozen=True)
class Parameters:
    sample_time: float = 0.05
    a_v: float = 0.0
    b_v: float = 1.0
    f_c: float = 0.0
    v_epsilon: float = 0.05
    g_gamma: float = 0.0
    a_r: float = 0.0
    b_r: float = 1.0
    v_max: float = 0.30
    r_max: float = 1.50
    kappa_max: float = 2.0
    dv_dt_max: float = 0.35
    dr_dt_max: float = 1.20
    tau_sum_max: float = 4.0
    tau_diff_max: float = 4.0
    tau_side_max: float = 3.0
    eta_tau: float = 0.90
    track_width: float = 0.72
    nonreversing_wheels: bool = False
    w_track_v: float = 10.0
    w_track_r: float = 10.0
    w_smooth_v: float = 2.0
    w_smooth_r: float = 2.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, bool) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ('sample_time', 'v_epsilon', 'v_max', 'r_max', 'kappa_max',
                     'dv_dt_max', 'dr_dt_max', 'tau_sum_max', 'tau_diff_max',
                     'tau_side_max', 'track_width', 'w_track_v', 'w_track_r'):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if abs(self.b_v) < 1e-12 or abs(self.b_r) < 1e-12:
            raise ValueError('actuation gains must be nonzero')
        if not 0 < self.eta_tau <= 1:
            raise ValueError('eta_tau must be in (0, 1]')
        if min(self.w_smooth_v, self.w_smooth_r) < 0:
            raise ValueError('smoothness weights must be nonnegative')


@dataclass(frozen=True)
class Result:
    status: str
    command: tuple | None
    torques: tuple | None
    max_residual: float | None
    constraints: tuple


def feasible(x, constraints, tol=1e-9):
    return all(a*x[0] + b*x[1] - c <= tol for a, b, c in constraints)


def solve_qp_2d(hv, hr, fv, fr, constraints):
    """Minimize .5*(hv*v²+hr*r²)+fv*v+fr*r subject to a*v+b*r<=c.

    Enumerates the interior optimum, weighted edge projections, and vertices.
    Returns None if no feasible candidate exists; never returns a violating
    unconstrained optimum as a successful solution.
    """
    constraints = tuple(constraints)
    if not all(math.isfinite(z) for z in (hv, hr, fv, fr)) or min(hv, hr) <= 0:
        raise ValueError('finite positive Hessian diagonal required')
    if any(not all(math.isfinite(z) for z in row) for row in constraints):
        raise ValueError('finite constraints required')
    v0, r0 = -fv/hv, -fr/hr
    candidates = [(v0, r0)]
    for a, b, c in constraints:
        den = a*a/hv + b*b/hr
        if den > 1e-24:
            lam = (a*v0+b*r0-c)/den
            candidates.append((v0-a*lam/hv, r0-b*lam/hr))
    for (a, b, c), (d, e, f) in combinations(constraints, 2):
        det = a*e-b*d
        scale = math.hypot(a,b)*math.hypot(d,e)
        if abs(det) > 1e-12*max(scale, 1e-24):
            candidates.append(((c*e-b*f)/det, (a*f-c*d)/det))
    valid = [x for x in candidates if feasible(x, constraints)]
    return min(valid, key=lambda x: .5*(hv*x[0]**2+hr*x[1]**2)+fv*x[0]+fr*x[1]) if valid else None


def torque_affine(state, p):
    """Return coefficients for summed and differential effort, with state fixed."""
    v, r = state
    s = (1/(p.sample_time*p.b_v), 0.0,
         (-v/p.sample_time+p.f_c*math.tanh(v/p.v_epsilon)+p.a_v*v+p.g_gamma)/p.b_v)
    d = (0.0, 1/(p.sample_time*p.b_r), (-r/p.sample_time+p.a_r*r)/p.b_r)
    return s, d


def torques(command, state, p):
    s, d = torque_affine(state, p)
    ts = s[0]*command[0]+s[1]*command[1]+s[2]
    td = d[0]*command[0]+d[1]*command[1]+d[2]
    return ts, td, .5*(ts-td), .5*(ts+td)  # sum, diff, left, right


def govern(nominal, state, previous, p=Parameters()):
    """Forward-motion formulation: 0<=v, |r|<=kappa_max*v.

    Slew bounds use the previous issued command; torque prediction uses the
    measured state. Model validity and actual traction are separate questions.
    """
    if not all(math.isfinite(z) for x in (nominal,state,previous) for z in x):
        raise ValueError('finite two-component inputs required')
    dv, dr = p.dv_dt_max*p.sample_time, p.dr_dt_max*p.sample_time
    vlo, vhi = max(0., previous[0]-dv), min(p.v_max, previous[0]+dv)
    rlo, rhi = max(-p.r_max, previous[1]-dr), min(p.r_max, previous[1]+dr)
    cons = [(1.,0.,vhi),(-1.,0.,-vlo),(0.,1.,rhi),(0.,-1.,-rlo),
            (-p.kappa_max,1.,0.),(-p.kappa_max,-1.,0.)]
    if p.nonreversing_wheels:
        cons.extend([(-1.,p.track_width/2,0.),(-1.,-p.track_width/2,0.)])
    s, d = torque_affine(state, p)
    left = tuple((x-y)/2 for x,y in zip(s,d))
    right = tuple((x+y)/2 for x,y in zip(s,d))
    for (a,b,c), limit in [(s,p.tau_sum_max),(d,p.tau_diff_max),
                            (left,p.tau_side_max),(right,p.tau_side_max)]:
        limit *= p.eta_tau
        cons.extend([(a,b,limit-c),(-a,-b,limit+c)])
    hv = 2*(p.w_track_v+p.w_smooth_v)
    hr = 2*(p.w_track_r+p.w_smooth_r)
    fv = -2*(p.w_track_v*nominal[0]+p.w_smooth_v*previous[0])
    fr = -2*(p.w_track_r*nominal[1]+p.w_smooth_r*previous[1])
    x = solve_qp_2d(hv,hr,fv,fr,cons)
    if x is None:
        return Result('infeasible',None,None,None,tuple(cons))
    residual = max(a*x[0]+b*x[1]-c for a,b,c in cons)
    return Result('optimal',x,torques(x,state,p),residual,tuple(cons))
