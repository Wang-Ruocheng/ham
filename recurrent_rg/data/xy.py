"""Continuous square-lattice XY model; embedded reflection multi-cluster sweeps.

H/J=-sum_<ij> cos(theta_i-theta_j), K=J/T. Angles remain continuous.
Periodic right/down links (including parallel L=2 links) are counted once.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def _sweep(theta, length, K):
    n = length*length
    normal = np.random.uniform(-np.pi, np.pi)
    projection = np.cos(theta-normal)
    parent = np.arange(n)
    for i in range(n):
        row, col = i//length, i%length
        for j in (row*length+(col+1)%length, ((row+1)%length)*length+col):
            product = projection[i]*projection[j]
            if product > 0 and np.random.random() < -np.expm1(-2*K*product):
                parent[_root(parent,i)] = _root(parent,j)
    flips = np.random.random(n) < .5
    for i in range(n):
        if flips[_root(parent,i)]:
            theta[i] = (2*normal+np.pi-theta[i]+np.pi) % (2*np.pi)-np.pi


@njit(cache=True)
def _sample(K, length, samples, seed, burn, spacing, ordered):
    np.random.seed(seed)
    theta = np.zeros(length*length) if ordered else np.random.uniform(-np.pi,np.pi,length*length)
    for _ in range(burn): _sweep(theta,length,K)
    result = np.empty((samples,length,length),np.float32)
    for t in range(samples):
        for _ in range(spacing): _sweep(theta,length,K)
        result[t] = theta.reshape(length,length)
    return result


def sample_xy(K, length, samples, seed, burn=2048, spacing=16, ordered=False):
    if not np.isfinite(K) or K < 0: raise ValueError('Finite nonnegative K required')
    if any(type(v) is not int for v in (length,samples,seed,burn,spacing)):
        raise ValueError('Integer counts required')
    if length < 2 or samples < 1 or burn < 0 or spacing < 1 or seed < 0:
        raise ValueError('Invalid size/count/seed')
    return _sample(float(K),length,samples,seed,burn,spacing,bool(ordered))


def wrap(x):
    return (np.asarray(x)+np.pi) % (2*np.pi)-np.pi


def observables(theta, K):
    """Per-configuration estimators; helicity requires ensemble current correction."""
    x = np.asarray(theta,dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != x.shape[2]: raise ValueError('Expected [samples,L,L]')
    n=x.shape[1]**2
    dx=wrap(np.roll(x,-1,2)-x); dy=wrap(np.roll(x,-1,1)-x)
    winding=(dx+np.roll(dy,-1,2)-np.roll(dx,-1,1)-dy)/(2*np.pi)
    charge=np.rint(winding)
    if np.max(abs(winding-charge))>1e-8 or np.max(abs(charge.sum((1,2))))>0:
        raise FloatingPointError('Invalid periodic vorticity')
    mx=np.cos(x).mean((1,2)); my=np.sin(x).mean((1,2))
    sx=np.sin(dx).sum((1,2)); sy=np.sin(dy).sum((1,2))
    cx=np.cos(dx).mean((1,2)); cy=np.cos(dy).mean((1,2))
    corr=np.stack([.5*(np.cos(x-np.roll(x,r,1)).mean((1,2))+
                            np.cos(x-np.roll(x,r,2)).mean((1,2))) for r in range(1,x.shape[1]//2+1)],1)
    return dict(energy=-(cx+cy), magnetization_squared=mx*mx+my*my,
                vortex_density=abs(charge).mean((1,2)), correlation=corr,
                helicity_raw=.5*(cx+cy-K*(sx*sx+sy*sy)/n),
                current_x=sx,current_y=sy)


def chain_diagnostics(obs, K, length):
    result={}
    for name in ('energy','magnetization_squared','vortex_density','helicity_raw'):
        v=np.asarray(obs[name]); centered=v-v.mean(); var=np.mean(centered**2)
        rho=np.array([np.mean(centered[:-k]*centered[k:])/var if var>1e-20 else 0.
                      for k in range(1,min(len(v)//2,64))])
        positive=[]
        for k in range(0,len(rho)-1,2):
            pair=rho[k]+rho[k+1]
            if pair<=0: break
            positive.append(pair)
        tau=max(1.,1+2*sum(positive))
        blocks=np.array_split(v,8); blockmeans=np.array([b.mean() for b in blocks])
        result[name]=dict(mean=float(v.mean()),lag1=float(rho[0]) if len(rho) else 0.,
            iat=tau,ess=float(len(v)/tau),batch_se=float(blockmeans.std(ddof=1)/np.sqrt(8)))
    correction=.5*K*(np.mean(obs['current_x'])**2+np.mean(obs['current_y'])**2)/(length**2)
    result['helicity_modulus_over_J']=float(np.mean(obs['helicity_raw'])+correction)
    result['correlation_mean']=obs['correlation'].mean(0).tolist()
    result['warning']=any(result[n]['ess']<32 for n in ('energy','magnetization_squared','vortex_density'))
    return result
