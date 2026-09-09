import argparse, os, math
from stitch_fss import first_root

p=argparse.ArgumentParser()
p.add_argument('--L',type=int,required=True)
p.add_argument('--T',type=float,required=True)
p.add_argument('--width',type=int,default=24,help='distance between umbrella centers in N_H units')
p.add_argument('--kappa',type=float,default=0.05)
p.add_argument('--replicas',type=int,default=2)
p.add_argument('--equil-sweeps',type=int,default=20)
p.add_argument('--meas-sweeps',type=int,default=120)
p.add_argument('--outdir',default='runs')
p.add_argument('--script',default='run_jobs.sh')
a=p.parse_args();N=a.L*a.L;mu=first_root(a.L,a.T)[0]
centers=list(range(0,N+1,a.width))
if centers[-1]!=N: centers.append(N)
os.makedirs(a.outdir,exist_ok=True)
with open(a.script,'w') as f:
    f.write('#!/usr/bin/env bash\nset -euo pipefail\n')
    for c in centers:
      for r in range(a.replicas):
        seed=100000*a.L+1000*int(round(1e5*a.T))+100*c+r
        out=os.path.join(a.outdir,f'L{a.L}_T{a.T:.5f}_c{c:04d}_r{r}.npz')
        f.write(f'python umbrella_window.py --L {a.L} --T {a.T} --mu {mu:.15g} --center {c} --kappa {a.kappa} --equil-sweeps {a.equil_sweeps} --meas-sweeps {a.meas_sweeps} --seed {seed} --out {out}\n')
os.chmod(a.script,0o755)
print('mu0',mu,'centers',centers,'jobs',len(centers)*a.replicas,'script',a.script)
