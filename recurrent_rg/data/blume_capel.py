"""Zero-field square Blume-Capel: H=-J sum s_i s_j + Delta sum s_i^2.

One hybrid sweep is N random-site heat-bath updates followed by a full
Swendsen-Wang sign update on the occupied lattice. Heat bath changes the
vacancies; sign clusters alone would not be ergodic. No physical labels
enter the QCR training objective.
"""
import numpy as np
from numba import njit
from .potts import _root


def conditional_probabilities(neighbor_sum, temperature, delta=1., exchange=1.):
    s=np.array([-1.,0.,1.])
    z=(exchange*np.asarray(neighbor_sum)[...,None]*s-delta*s*s)/temperature
    z-=z.max(axis=-1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(axis=-1,keepdims=True)


@njit(cache=True)
def _sweep(spins, length, probabilities, bond_probability):
    n=length*length
    for _ in range(n):
        i=np.random.randint(n);row,col=i//length,i%length
        h=(spins[row*length+(col+1)%length]+spins[row*length+(col-1)%length]
           +spins[((row+1)%length)*length+col]+spins[((row-1)%length)*length+col])
        u=np.random.random();p=probabilities[h+4]
        spins[i]=-1 if u<p[0] else (0 if u<p[0]+p[1] else 1)
    parent=np.arange(n)
    for i in range(n):
        if spins[i]==0:continue
        row,col=i//length,i%length
        for j in (row*length+(col+1)%length,((row+1)%length)*length+col):
            if spins[i]==spins[j] and np.random.random()<bond_probability:
                parent[_root(parent,i)]=_root(parent,j)
    signs=2*np.random.randint(0,2,n)-1
    for i in range(n):
        if spins[i]!=0:spins[i]=signs[_root(parent,i)]


@njit(cache=True)
def _sample(length,samples,seed,burn,spacing,start,probabilities,bond_probability):
    np.random.seed(seed)
    spins=np.random.randint(0,3,length*length)-1 if start==2 else np.full(length*length,start,np.int64)
    for _ in range(burn):_sweep(spins,length,probabilities,bond_probability)
    result=np.empty((samples,length,length),np.int8)
    for i in range(samples):
        for _ in range(spacing):_sweep(spins,length,probabilities,bond_probability)
        result[i]=spins.reshape(length,length)
    return result


def sample_blume_capel(temperature,length,samples,seed,*,delta=1.,exchange=1.,burn=1024,spacing=8,start='random'):
    """Return raw {-1,0,+1} MCMC spins on a periodic LxL lattice.

    L=2 uses the periodic multigraph (two bonds per nearest-neighbor pair),
    matching the right/down-bond energy convention in observables.
    """
    if not np.isfinite(temperature) or temperature<=0 or not np.isfinite(delta) or not np.isfinite(exchange) or exchange<0:
        raise ValueError('Finite T>0, Delta and ferromagnetic J>=0 required')
    if any(not isinstance(v,(int,np.integer)) for v in (length,samples,seed,burn,spacing)) or length<2 or samples<1 or burn<0 or spacing<1 or not 0<=seed<2**32:
        raise ValueError('Invalid lattice, sample count, seed or sweep count')
    starts={'random':2,'plus':1,'minus':-1,'zero':0}
    if start not in starts:raise ValueError('Unknown initial state')
    p=conditional_probabilities(np.arange(-4,5),temperature,delta,exchange)
    return _sample(length,samples,seed,burn,spacing,starts[start],p,-np.expm1(-2*exchange/temperature))


def observable_series(x,delta=1.,exchange=1.):
    x=np.asarray(x,dtype=np.float64)
    m=x.mean((1,2));rho=(x*x).mean((1,2))
    bonds=(x*np.roll(x,1,axis=1)+x*np.roll(x,1,axis=2)).mean((1,2))
    return dict(energy=-exchange*bonds+delta*rho,magnetization=m,
                abs_magnetization=np.abs(m),occupancy=rho,magnetization_squared=m*m)


def sampling_observables(x,delta=1.):
    result={}
    for name,v in observable_series(x,delta).items():
        blocks=[b.mean() for b in np.array_split(v,min(16,len(v))) if len(b)]
        lag=float(np.corrcoef(v[:-1],v[1:])[0,1]) if len(v)>2 and v.std()>1e-12 else None
        result[name]=dict(mean=float(v.mean()),block_mean_se=float(np.std(blocks,ddof=1)/np.sqrt(len(blocks))) if len(blocks)>1 else None,lag1=lag)
    return result
