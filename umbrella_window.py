import argparse, math, time, os
import numpy as np
from scipy.linalg import eigh
from stitch_fss import first_root
from fermion_core import diagonalize_state, gram_dense, omega_from_lam, delta_omega_matsubara


def harmonic_bias(nh, center, kappa, N):
    # Dimensionless shape is stable across sizes because distance is in fraction x_H.
    x=nh/N; xc=center/N
    return 0.5*kappa*N*(x-xc)**2


def make_initial(L, center, rng):
    N=L*L; nh=int(round(center)); nh=max(0,min(N,nh))
    typ=np.zeros(N,dtype=np.int8)
    if nh: typ[rng.choice(N,size=nh,replace=False)]=1
    return typ.reshape((L,L))


def main():
    p=argparse.ArgumentParser(description='Direct-fermion harmonic umbrella window in N_H')
    p.add_argument('--L',type=int,required=True)
    p.add_argument('--T',type=float,required=True)
    p.add_argument('--mu',type=float,default=None,help='reference chemical potential; default homogeneous crossing')
    p.add_argument('--center',type=float,required=True,help='umbrella center in N_H units')
    p.add_argument('--kappa',type=float,default=0.05,help='bias strength in J; U=0.5*kappa*N*(x-xc)^2')
    p.add_argument('--equil-sweeps',type=int,default=20)
    p.add_argument('--meas-sweeps',type=int,default=120)
    p.add_argument('--thin',type=int,default=4,help='record every this many proposals after equilibration')
    p.add_argument('--nfreq',type=int,default=128)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    N=a.L*a.L; mu=first_root(a.L,a.T)[0] if a.mu is None else a.mu
    rng=np.random.default_rng(a.seed)
    typ=make_initial(a.L,a.center,rng)
    G,lam,Q=diagonalize_state(a.L,typ); Om=omega_from_lam(lam,a.T,mu)
    total=(a.equil_sweeps+a.meas_sweeps)*N; burn=a.equil_sweeps*N
    nhs=[]; oms=[]; lams=[]; acc=0; att=0; t0=time.time()
    for step in range(total):
        i,j=int(rng.integers(a.L)),int(rng.integers(a.L)); n0=int(typ.sum())
        prop=typ.copy(); prop[i,j]^=1; n1=int(prop.sum())
        Gp=gram_dense(a.L,prop)
        dOm=delta_omega_matsubara(lam,Q,G,Gp,a.T,mu,a.nfreq)
        dU=harmonic_bias(n1,a.center,a.kappa,N)-harmonic_bias(n0,a.center,a.kappa,N)
        loga=-(dOm+dU)/a.T; att+=1
        if loga>=0 or math.log(rng.random())<loga:
            typ=prop;G=Gp;lam,Q=eigh(G,check_finite=False,driver='evd');Om+=dOm;acc+=1
        if step>=burn and (step-burn)%a.thin==0:
            nhs.append(int(typ.sum())); oms.append(float(Om)); lams.append(lam.copy())
        if (step+1)%(10*N)==0:
            print(f'L={a.L} center={a.center:g} sweep={(step+1)/N:.1f} nH={int(typ.sum())} acc={acc/att:.3f} elapsed={time.time()-t0:.1f}s',flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True)
    np.savez_compressed(a.out,nH=np.asarray(nhs,dtype=np.int16),Omega=np.asarray(oms),lam=np.asarray(lams),
        L=a.L,T=a.T,mu=mu,center=a.center,kappa=a.kappa,equil_sweeps=a.equil_sweeps,meas_sweeps=a.meas_sweeps,
        thin=a.thin,nfreq=a.nfreq,seed=a.seed,acceptance=acc/max(att,1),elapsed=time.time()-t0)
    print('saved',a.out,'samples',len(nhs),'acceptance',acc/max(att,1))

if __name__=='__main__': main()
