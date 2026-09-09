import argparse, glob, math, json
import numpy as np
from scipy.optimize import brentq
from pymbar import MBAR
from fermion_core import omega_from_lam


def bias(nh, center, kappa, N):
    x=nh/N;xc=center/N
    return 0.5*kappa*N*(x-xc)**2


def logsumexp(a):
    m=np.max(a); return m+np.log(np.sum(np.exp(a-m)))


def main():
 p=argparse.ArgumentParser()
 p.add_argument('pattern',help='glob for umbrella npz files of one L,T')
 p.add_argument('--mu-span',type=float,default=0.02)
 p.add_argument('--bins',type=int,default=None)
 p.add_argument('--out',default='mbar_summary.json')
 a=p.parse_args();files=sorted(glob.glob(a.pattern))
 if not files: raise SystemExit('no files')
 dat=[np.load(f,allow_pickle=False) for f in files]
 L=int(dat[0]['L']);T=float(dat[0]['T']);mu0=float(dat[0]['mu']);N=L*L
 # Pool independent replicas that share the same umbrella Hamiltonian into one MBAR state.
 keys=[]
 for d in dat:
     key=(float(d['center']),float(d['kappa']))
     if key not in keys: keys.append(key)
 centers=np.array([k[0] for k in keys]); kappas=np.array([k[1] for k in keys])
 nH_parts=[]; lam_parts=[]; N_k=[]
 for key in keys:
     ds=[d for d in dat if (float(d['center']),float(d['kappa']))==key]
     nH_parts.append(np.concatenate([d['nH'] for d in ds]).astype(int))
     lam_parts.append(np.concatenate([d['lam'] for d in ds],axis=0))
     N_k.append(sum(len(d['nH']) for d in ds))
 nH=np.concatenate(nH_parts); lams=np.concatenate(lam_parts,axis=0); N_k=np.asarray(N_k,dtype=int)
 Om0=np.array([omega_from_lam(x,T,mu0) for x in lams])
 K=len(keys);M=len(nH);u_kn=np.empty((K,M))
 for k in range(K): u_kn[k]= (Om0 + np.array([bias(n,centers[k],kappas[k],N) for n in nH]))/T
 mbar=MBAR(u_kn,N_k,verbose=False,relative_tolerance=1e-10,maximum_iterations=10000)
 f_k=np.asarray(mbar.f_k)
 logden=np.full(M,-np.inf)
 # log denominator: log sum_k N_k exp(f_k-u_kn)
 terms=[]
 for k in range(K):
     if N_k[k]>0: terms.append(np.log(N_k[k])+f_k[k]-u_kn[k])
 A=np.vstack(terms);mx=np.max(A,axis=0);logden=mx+np.log(np.sum(np.exp(A-mx),axis=0))
 def weights(mu):
     Om=np.array([omega_from_lam(x,T,mu) for x in lams])
     lw=-Om/T-logden; lw-=logsumexp(lw); return np.exp(lw)
 def phase_balance(mu):
     w=weights(mu);m=2*nH/N-1
     return w[m>0].sum()-w[m<0].sum()
 lo,hi=mu0-a.mu_span,mu0+a.mu_span
 flo,fhi=phase_balance(lo),phase_balance(hi)
 if flo*fhi>0:
     raise SystemExit(f'equal-weight root not bracketed in [{lo},{hi}]: balances {flo},{fhi}; increase --mu-span')
 mueq=brentq(phase_balance,lo,hi,xtol=1e-12);w=weights(mueq);m=2*nH/N-1
 mean=lambda x: float(np.sum(w*x))
 m2=mean(m*m);m4=mean(m**4);binder=1-m4/(3*m2*m2);chi=N/T*(m2-mean(m)**2)
 # weighted discrete P(nH)
 P=np.bincount(nH,weights=w,minlength=N+1);P=P/P.sum()
 # homology densities from established exact relation for this periodic construction
 b2=np.where(nH>0,nH,1); b1=np.where(nH>0,2*N+1+nH,2*N+2)
 result={'L':L,'T':T,'mu0':mu0,'mu_eq':mueq,'n_files':len(files),'n_windows':K,'n_samples':M,
         'mean_abs_m':mean(np.abs(m)),'mean_m':mean(m),'binder':binder,'chi_m':chi,'chi_m_T_over_N':chi*T/N,
         'mean_b1_over_N':mean(b1/N),'mean_b2_over_N':mean(b2/N),
         'phase_weight_pos':float(w[m>0].sum()),'phase_weight_neg':float(w[m<0].sum()),
         'P_nH':P.tolist(),'centers':centers.tolist(),'N_k':N_k.tolist()}
 with open(a.out,'w') as f: json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='P_nH'},indent=2))

if __name__=='__main__': main()
