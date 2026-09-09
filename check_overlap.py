import argparse,glob,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('pattern');p.add_argument('--out',default='overlap.json');a=p.parse_args()
files=sorted(glob.glob(a.pattern)); dat=[]
for f in files:
 d=np.load(f); dat.append((f,float(d['center']),np.asarray(d['nH'],int)))
dat.sort(key=lambda x:x[1]); rows=[]; ok=True
for A,B in zip(dat[:-1],dat[1:]):
 sa=set(A[2].tolist());sb=set(B[2].tolist());inter=sorted(sa&sb)
 frac=min(len(inter)/max(len(sa),1),len(inter)/max(len(sb),1)); passed=len(inter)>=3
 ok &= passed;rows.append({'left':A[0],'right':B[0],'centers':[A[1],B[1]],'shared_bins':inter,'n_shared':len(inter),'pass':passed})
out={'pass_all_neighbors':bool(ok),'pairs':rows}
with open(a.out,'w') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
