import argparse, json, time
import numpy as np
from scipy.linalg import eigh
from stitch_fss import first_root
from fermion_core import diagonalize_state, gram_dense, omega_from_lam, delta_omega_matsubara


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--L',type=int,default=16)
    p.add_argument('--T',type=float,default=0.0038)
    p.add_argument('--mu',type=float,default=None)
    p.add_argument('--moves',type=int,default=24)
    p.add_argument('--nfreq',type=int,default=128)
    p.add_argument('--seed',type=int,default=20260831)
    p.add_argument('--out',default='validation.json')
    a=p.parse_args()
    mu = first_root(a.L,a.T)[0] if a.mu is None else a.mu
    rng=np.random.default_rng(a.seed)
    typ=rng.integers(0,2,size=(a.L,a.L),dtype=np.int8)
    G,lam,Q=diagonalize_state(a.L,typ)
    Om=omega_from_lam(lam,a.T,mu)
    errs=[]; rows=[]
    for s in range(a.moves):
        i,j=int(rng.integers(a.L)),int(rng.integers(a.L))
        prop=typ.copy(); prop[i,j]^=1
        Gp=gram_dense(a.L,prop)
        t0=time.time(); da=delta_omega_matsubara(lam,Q,G,Gp,a.T,mu,a.nfreq); t1=time.time()
        lamp,Qp=eigh(Gp,check_finite=False,driver='evd')
        Omp=omega_from_lam(lamp,a.T,mu); t2=time.time()
        de=Omp-Om; err=da-de; errs.append(err)
        rows.append({'move':s,'nH':int(typ.sum()),'delta_exact':float(de),'delta_lowrank':float(da),'error':float(err),'t_lowrank':t1-t0,'t_diag':t2-t1})
        if rng.random()<0.5:
            typ=prop;G=Gp;lam=lamp;Q=Qp;Om=Omp
    out={'L':a.L,'T':a.T,'mu':mu,'nfreq':a.nfreq,'moves':a.moves,
         'max_abs_error':float(np.max(np.abs(errs))),
         'rms_error':float(np.sqrt(np.mean(np.square(errs)))),'rows':rows}
    with open(a.out,'w') as f: json.dump(out,f,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
    if out['max_abs_error']>1e-5:
        raise SystemExit('FAIL: increase --nfreq until max_abs_error < 1e-5 J')

if __name__=='__main__': main()
