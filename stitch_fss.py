import numpy as np, math, itertools, json, time
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.linalg import eigvalsh

FH=[(1,2,4),(1,2,5),(1,2,6),(1,3,4),(1,3,5),(2,3,4),(2,3,5),(4,5,6)]
FL=[(1,2,3),(1,2,4),(1,2,5),(1,3,4),(1,5,6),(2,3,5),(2,4,6),(3,4,5)]
label_to_role={1:6,2:3,3:2,4:4,5:1,6:5}
role_coord={1:(0,0),2:(1,0),3:(1,1),4:(0,1)}
# translational pattern
vdesc={5:('c',(0,0)),3:('c',(1,0)),4:('c',(0,1)),2:('c',(1,1)),6:('a',(0,0)),1:('b',(0,0))}
edge_types=[]
for d in [(1,0),(0,1),(1,1),(-1,1)]: edge_types.append(('cc',d))
for p in ['a','b']:
    for d in [(0,0),(1,0),(0,1),(1,1)]: edge_types.append((p+'c',d))
edge_types.append(('ba',(0,0)))
eind={e:i for i,e in enumerate(edge_types)}
allowed_cc={(1,0),(0,1),(1,1),(-1,1)}

def canonical_directed_edge(lu,lv):
    ou,tu=vdesc[lu]; ov,tv=vdesc[lv]
    tu=np.array(tu); tv=np.array(tv)
    if ou=='c' and ov=='c':
        d=tuple(tv-tu)
        if d in allowed_cc: et=('cc',d); anchor=tuple(tu); sign=1
        elif tuple(-np.array(d)) in allowed_cc: et=('cc',tuple(-np.array(d))); anchor=tuple(tv); sign=-1
        else: raise RuntimeError
    elif ou in ['a','b'] and ov=='c': et=(ou+'c',tuple(tv-tu)); anchor=tuple(tu); sign=1
    elif ou=='c' and ov in ['a','b']: et=(ov+'c',tuple(tu-tv)); anchor=tuple(tv); sign=-1
    elif {ou,ov}=={'a','b'}: et=('ba',(0,0)); anchor=(0,0); sign=1 if (ou,ov)==('b','a') else -1
    else: raise RuntimeError
    return eind[et],anchor,sign

def bloch_B1(kx,ky):
    z=lambda d: np.exp(-1j*(kx*d[0]+ky*d[1]))
    B=np.zeros((3,13),complex)
    for idx,(kind,d) in enumerate(edge_types):
        if kind=='cc': B[0,idx]=-1+z(d)
        elif kind=='ac': B[1,idx]=-1; B[0,idx]=z(d)
        elif kind=='bc': B[2,idx]=-1; B[0,idx]=z(d)
        else: B[2,idx]=-1; B[1,idx]=1
    return B

def bloch_B2(kx,ky,faces):
    z=lambda d: np.exp(-1j*(kx*d[0]+ky*d[1]))
    B=np.zeros((13,8),complex)
    for col,(i,j,k) in enumerate(faces):
        for (u,v),coef in [((j,k),1),((i,k),-1),((i,j),1)]:
            ei,s,sg=canonical_directed_edge(u,v)
            B[ei,col]+=coef*sg*z(s)
    return B

def precompute_hom(L,faces):
    s1=[];s2=[]
    for nx in range(L):
      kx=2*np.pi*nx/L
      for ny in range(L):
        ky=2*np.pi*ny/L
        s1.extend(np.linalg.svd(bloch_B1(kx,ky),compute_uv=False))
        s2.extend(np.linalg.svd(bloch_B2(kx,ky,faces),compute_uv=False))
    return np.array(s1),np.array(s2)

def pair_g(s,T,mu):
    return -T*(np.logaddexp(0,-(s-mu)/T)+np.logaddexp(0,-(-s-mu)/T))

def omega_hom(s1,s2,T,mu,N):
    # 2N extra exact zero modes not captured by paired singular values
    g0=-T*np.logaddexp(0,mu/T)
    return pair_g(s1,T,mu).sum()+pair_g(s2,T,mu).sum()+2*N*g0

def density_hom(s1,s2,T,mu,N):
    # number per cell from all states; f(E)=1/(exp((E-mu)/T)+1)
    def f(e):
        x=(e-mu)/T
        # stable
        return np.where(x>0,np.exp(-x)/(1+np.exp(-x)),1/(1+np.exp(x)))
    num=(f(s1)+f(-s1)).sum()+(f(s2)+f(-s2)).sum()+2*N*f(0.0)
    return num/N

def build_B2_sparse(L,types):
    N=L*L
    corner=lambda i,j:(i%L)*L+(j%L)
    def priv(i,j,r):
        cell=(i%L)*L+(j%L)
        return N+2*cell+(0 if r==5 else 1)
    def rolevid(i,j,r):
        if r<=4:
            di,dj=role_coord[r]; return corner(i+di,j+dj)
        return priv(i,j,r)
    def labelvid(i,j,l): return rolevid(i,j,label_to_role[l])
    edges=set()
    for i in range(L):
      for j in range(L):
        vs=[labelvid(i,j,l) for l in range(1,7)]
        for a in range(6):
          for b in range(a+1,6): edges.add(tuple(sorted((vs[a],vs[b]))))
    edges=sorted(edges); eidx={e:k for k,e in enumerate(edges)}
    rows=[];cols=[];data=[]; col=0
    for i in range(L):
      for j in range(L):
        fs=FH if types[i,j] else FL
        for (a,b,c) in fs:
          va,vb,vc=labelvid(i,j,a),labelvid(i,j,b),labelvid(i,j,c)
          for (x,y),coef in [((vb,vc),1),((va,vc),-1),((va,vb),1)]:
            if x<y: e=(x,y); sg=1
            else: e=(y,x); sg=-1
            rows.append(eidx[e]);cols.append(col);data.append(coef*sg)
          col+=1
    B2=coo_matrix((data,(rows,cols)),shape=(len(edges),8*N),dtype=float).tocsr()
    return B2

def svals_config(L,types):
    B=build_B2_sparse(L,types)
    G=(B.T@B).toarray()
    lam=eigvalsh(G,check_finite=False,overwrite_a=True,driver='evd')
    return np.sqrt(np.clip(lam,0,None))

def omega_s2(sv,T,mu): return pair_g(sv,T,mu).sum()

def first_root(L,T):
    s1,sH=precompute_hom(L,FH); _,sL=precompute_hom(L,FL); N=L*L
    f=lambda mu:(omega_hom(s1,sH,T,mu,N)-omega_hom(s1,sL,T,mu,N))/N
    return brentq(f,-2.65,-2.2),s1,sH,sL

def interface_sigma(L,T,mu,orient):
    # half H half L stripe; compare to homogeneous average at exact same finite L crossing
    typ=np.zeros((L,L),int)
    if orient=='x': typ[:L//2,:]=1
    else: typ[:,:L//2]=1
    sv=svals_config(L,typ)
    # only B2-dependent contribution; homogeneous crossing B2 components equal at mu, so avg same either
    _,sH=precompute_hom(L,FH); _,sL=precompute_hom(L,FL)
    Om=omega_s2(sv,T,mu)
    Oh=omega_s2(sH,T,mu); Ol=omega_s2(sL,T,mu)
    return (Om-0.5*(Oh+Ol))/(2*L)

def defect_energy(L,T,mu,baseH=False):
    typ=np.ones((L,L),int) if baseH else np.zeros((L,L),int)
    typ[0,0]=0 if baseH else 1
    sv=svals_config(L,typ)
    _,sH=precompute_hom(L,FH);_,sL=precompute_hom(L,FL)
    ob=omega_s2(sH if baseH else sL,T,mu)
    return omega_s2(sv,T,mu)-ob

if __name__=='__main__':
    temps=[0.0025,0.0033,0.0038,0.0042,0.0050]
    Ls=[6,8,10,12,16]
    out={'homogeneous':{},'interfaces':{},'defects':{}}
    for T in temps:
      out['homogeneous'][str(T)]={}
      for L in Ls:
        root,s1,sH,sL=first_root(L,T);N=L*L
        out['homogeneous'][str(T)][str(L)]={'mu_c':root,'rho_H':density_hom(s1,sH,T,root,N),'rho_L':density_hom(s1,sL,T,root,N)}
        print('hom',T,L,root)
    # interfaces and defects at 3 temperatures + endpoint-ish, finite L sizes (expensive for 16)
    for T in [0.0025,0.0033,0.0038,0.0042,0.0050]:
      out['interfaces'][str(T)]={}; out['defects'][str(T)]={}
      for L in [6,8,10,12,16]:
        mu=out['homogeneous'][str(T)][str(L)]['mu_c']
        t0=time.time(); sx=interface_sigma(L,T,mu,'x'); sy=interface_sigma(L,T,mu,'y')
        dHL=defect_energy(L,T,mu,False); dLH=defect_energy(L,T,mu,True)
        out['interfaces'][str(T)][str(L)]={'sigma_x':sx,'sigma_y':sy}
        out['defects'][str(T)][str(L)]={'H_in_L':dHL,'L_in_H':dLH}
        print('int',T,L,sx,sy,dHL,dLH,'dt',time.time()-t0,flush=True)
    print(json.dumps(out,indent=2))
    with open('/tmp/stitch_fss_results.json','w') as f: json.dump(out,f,indent=2)
