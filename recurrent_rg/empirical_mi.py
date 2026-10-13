"""Exact finite-cache binary MI using compressed conditional distributions.

No environment projection: merge E categories only if their empirical P(V|E)
is identical. Keep ensemble-dependent weights and marginals separate.
"""
from dataclasses import dataclass, field, replace
import numpy as np
import torch
from .rsmi import region_offsets


def binary_entropy(p):
    p=p.clamp(0,1);tiny=torch.finfo(p.dtype).tiny
    return -p*p.clamp_min(tiny).log()-(1-p)*(1-p).clamp_min(tiny).log()


def categorical_entropy(p):
    return -(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)


@dataclass
class _EmpiricalMISelection:
    """CPU row maps and canonical tensors derived from the static histogram.

    This is execution state, never part of the on-disk statistics cache. Sparse
    rows are already sorted/coalesced, so selected batches need no coalescing
    or device-to-host reads in the training loop.
    """
    weight_offsets: np.ndarray
    weight_conditionals: np.ndarray
    weights: torch.Tensor
    conditional: torch.Tensor
    conditional_offsets: np.ndarray | None = None
    conditional_columns: np.ndarray | None = None


def _selected_row_entries(offsets, rows):
    """Vectorized positions in ragged rows, retaining the requested row order."""
    lengths = offsets[rows + 1] - offsets[rows]
    ends = np.cumsum(lengths)
    starts = np.r_[0, ends[:-1]]
    positions = (np.repeat(offsets[rows] - starts, lengths)
                 + np.arange(int(ends[-1]) if len(ends) else 0))
    return positions, np.repeat(np.arange(len(rows)), lengths)


@dataclass
class EmpiricalBinaryMI:
    conditional: torch.Tensor
    weights: torch.Tensor
    marginal: torch.Tensor
    diagnostics: dict
    _selection: _EmpiricalMISelection | None = field(default=None, init=False,
                                                    repr=False, compare=False)

    def __len__(self):return len(self.marginal)

    def to(self, device, *, dtype=None, non_blocking=False):
        """Move cached statistics once, preserving empirical counts and sparsity.

        Histogram construction remains on CPU. The returned object can evaluate
        the same differentiable kernel objective on the kernel's device without
        repeatedly moving statistics inside the training loop.
        """
        move = lambda tensor: tensor.to(device=device, dtype=dtype,
                                        non_blocking=non_blocking)
        result = EmpiricalBinaryMI(move(self.conditional), move(self.weights),
                                   move(self.marginal), self.diagnostics)
        if self._selection is not None:
            selection = self._selection
            # Coalesced inputs share their already-moved public tensors. Direct
            # callers may have supplied uncoalesced tensors; retain the prepared
            # canonical version without repeating coalescing inside each step.
            result._selection = replace(selection,
                weights=(result.weights if selection.weights is self.weights
                         else move(selection.weights)),
                conditional=(result.conditional if selection.conditional is self.conditional
                             else move(selection.conditional)))
        return result

    def prepare_selection(self):
        """Index ensemble supports once on CPU before moving statistics.

        The original histogram tensors and historical cache payload are kept
        intact. Only row pointers/columns are indexed; no kernel, randomness,
        or ensemble weights enter this setup. Call ``information_for`` with
        the existing CPU-generated ensemble IDs after ``to(device)``.
        """
        if self._selection is not None:
            return self
        if self.weights.device.type != 'cpu' or self.conditional.device.type != 'cpu':
            raise ValueError('Prepare empirical MI selection on CPU before to(device)')
        weights = self.weights.to_sparse_coo().coalesce()
        weight_indices = weights.indices().numpy()
        weight_offsets = np.r_[0, np.cumsum(np.bincount(
            weight_indices[0], minlength=len(self)))]
        conditional = self.conditional
        conditional_offsets = conditional_columns = None
        if conditional.layout != torch.strided:
            conditional = conditional.to_sparse_coo().coalesce()
            conditional_indices = conditional.indices().numpy()
            conditional_offsets = np.r_[0, np.cumsum(np.bincount(
                conditional_indices[0], minlength=conditional.shape[0]))]
            # Own just the column map so moving the histogram can release its
            # larger CPU COO index storage (including the redundant row map).
            conditional_columns = conditional_indices[1].copy()
        self._selection = _EmpiricalMISelection(
            weight_offsets, weight_indices[1].copy(), weights, conditional,
            conditional_offsets, conditional_columns)
        return self

    def information_for(self, table, indices):
        """Exact MI for selected ensembles, in order and with repeats preserved.

        CPU selection IDs avoid a CUDA synchronization. One batch evaluates
        the union of active conditional rows and selected marginal rows, then
        applies the original per-ensemble weights. Setup and evaluation consume
        no random draws, and the support index is bounded by the histogram size.
        """
        if isinstance(indices, torch.Tensor):
            if indices.device.type != 'cpu':
                raise ValueError('Selected empirical MI indices must remain on CPU')
            if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                                     torch.uint8):
                raise TypeError('Selected empirical MI indices must be integers')
            indices = indices.numpy()
        ids = np.asarray(indices)
        if ids.ndim != 1:
            raise ValueError('Selected empirical MI indices must be one-dimensional')
        if ids.size and not np.issubdtype(ids.dtype, np.integer):
            raise TypeError('Selected empirical MI indices must be integers')
        ids = ids.astype(np.int64, copy=False)
        if np.any(ids < -len(self)) or np.any(ids >= len(self)):
            raise IndexError('Selected empirical MI ensemble index is out of range')
        ids = np.where(ids < 0, ids + len(self), ids)
        if self._selection is None:
            self.prepare_selection()
        selection = self._selection
        device = table.device
        # Keep all host-generated index copies nonblocking on CUDA. Supplying
        # device directly to as_tensor would use blocking copies by default.
        index_tensor = lambda values: torch.as_tensor(values, dtype=torch.long).to(
            device=device, non_blocking=True)
        marginal = self.marginal.index_select(0, index_tensor(ids))
        binary = table.shape[1] == 2
        marginal_entropy = (binary_entropy(marginal @ table[:, 1]) if binary
                            else categorical_entropy(marginal @ table))
        if not len(ids):
            return marginal_entropy

        weight_positions, owners = _selected_row_entries(selection.weight_offsets, ids)
        active, inverse = np.unique(selection.weight_conditionals[weight_positions],
                                    return_inverse=True)
        if selection.conditional_offsets is None:
            conditional = selection.conditional.index_select(0, index_tensor(active))
        else:
            positions, rows = _selected_row_entries(selection.conditional_offsets, active)
            conditional = torch.sparse_coo_tensor(
                index_tensor(np.stack((rows, selection.conditional_columns[positions]))),
                selection.conditional.values().index_select(0, index_tensor(positions)),
                (len(active), selection.conditional.shape[1]), device=device,
                is_coalesced=True)
        probabilities = table[:, 1:2] if binary else table
        conditional_probabilities = (torch.sparse.mm(conditional, probabilities)
                                     if conditional.is_sparse else conditional @ probabilities)
        conditional_entropy = (binary_entropy(conditional_probabilities[:, 0]) if binary
                               else categorical_entropy(conditional_probabilities))
        weights = torch.sparse_coo_tensor(
            index_tensor(np.stack((owners, inverse))),
            selection.weights.values().index_select(0, index_tensor(weight_positions)),
            (len(ids), len(active)), device=device, is_coalesced=True)
        return marginal_entropy - torch.sparse.mm(weights, conditional_entropy[:, None]).squeeze(-1)

    def information(self,table):
        if table.shape[1]!=2:
            hc=categorical_entropy(self.conditional@table)
            return categorical_entropy(self.marginal@table)-torch.sparse.mm(self.weights,hc[:,None]).squeeze(-1)
        positive=table[:,1]
        conditional_positive=(torch.sparse.mm(self.conditional,positive[:,None]).squeeze(-1)
                              if self.conditional.is_sparse else self.conditional@positive)
        conditional_entropy=binary_entropy(conditional_positive)
        return binary_entropy(self.marginal@positive)-torch.sparse.mm(self.weights,conditional_entropy[:,None]).squeeze(-1)


def binary_lattice_mi_statistics(x,*,flip=False,buffer=1,block=2):
    return lattice_mi_statistics(x,states=2,flip=flip,buffer=buffer,block=block)


def categorical_lattice_mi_statistics(x,*,flip=False,buffer=1,block=2):
    if flip:raise ValueError('Binary flip is not a categorical augmentation')
    return lattice_mi_statistics(x,states=x.shape[-1],buffer=buffer,block=block)


def lattice_mi_statistics(x,*,states,flip=False,buffer=1,shuffle_visible_seed=None,sparse_conditionals=None,environment_offsets=None,block=2):
    """All samples and translated V/E regions; individual ensembles preserved."""
    dimension=x.ndim-3
    visible,environment=region_offsets(dimension,block=block,buffer=buffer)
    if environment_offsets is not None:
        requested=torch.as_tensor(environment_offsets,dtype=torch.long)
        allowed={tuple(v) for v in environment.tolist()}
        if requested.ndim!=2 or requested.shape[1]!=dimension or len(requested)<1 or len({tuple(v) for v in requested.tolist()})!=len(requested) or any(tuple(v) not in allowed for v in requested.tolist()):
            raise ValueError('Environment subset must contain distinct sites from the buffered shell')
        environment=requested
    if any(n<block+2*(buffer+1) for n in x.shape[2:-1]):raise ValueError('Invalid lattice shape')
    if states**len(environment)>=2**62:raise ValueError('Environment code exceeds supported integer width')
    if x.shape[-1]==1:
        if states!=2 or not torch.all((x==1)|(x==-1)):raise ValueError('Expected binary +/-1 data')
        colors=(x[...,0]>0).long()
    else:
        if x.shape[-1]!=states or not torch.all((x==0)|(x==1)) or not torch.all(x.sum(-1)==1):raise ValueError('Expected one-hot data')
        colors=x.argmax(-1)
    nv=states**len(visible);ne=states**len(environment)
    sparse_conditionals=nv>256 if sparse_conditionals is None else sparse_conditionals
    signatures={};sparse_rows=[];sparse_cols=[];sparse_values=[]
    primitive=[];weight=[];owners=[];marginals=[];counts=[];singletons=[]
    shuffle_rng=np.random.default_rng(shuffle_visible_seed) if shuffle_visible_seed is not None else None
    for a,ensemble in enumerate(colors.cpu()):
        axes=tuple(range(1,dimension+1))
        def code(offsets):
            z=torch.zeros_like(ensemble,dtype=torch.long)
            for i,offset in enumerate(offsets.tolist()):
                z+=torch.roll(ensemble,tuple(-n for n in offset),axes)*states**i
            return z.flatten().numpy()
        v,e=code(visible),code(environment)
        if shuffle_rng is not None:v=v[shuffle_rng.permutation(len(v))]
        if flip:v,e=np.r_[v,nv-1-v],np.r_[e,ne-1-e]
        ue,inverse=np.unique(e,return_inverse=True)
        if sparse_conditionals:
            # Encode observed (environment index, visible code), never allocate
            # number_of_environments x q**visible_sites dense count tables.
            pairs,pair_counts=np.unique(inverse*nv+v,return_counts=True)
            erows=pairs//nv;vcols=pairs%nv
            starts=np.r_[0,np.flatnonzero(np.diff(erows))+1,len(pairs)]
            totals=np.bincount(inverse);counts.append(len(ue));singletons.append(float(np.mean(totals==1)))
            marginals.append(np.bincount(v,minlength=nv)/len(v))
            local={}
            for left,right in zip(starts[:-1],starts[1:]):
                cols=vcols[left:right];values=pair_counts[left:right]
                reduced=values//np.gcd.reduce(values)
                key=tuple(zip(cols.tolist(),reduced.tolist()))
                index=signatures.get(key)
                if index is None:
                    index=len(signatures);signatures[key]=index
                    sparse_rows.extend([index]*len(cols));sparse_cols.extend(cols.tolist())
                    sparse_values.extend((reduced/reduced.sum()).tolist())
                local[index]=local.get(index,0.)+float(values.sum()/len(v))
            owners.append(np.full(len(local),a));weight.append(np.array(list(local.values())))
            primitive.append(np.array(list(local),dtype=np.int64))
            continue
        hist=np.bincount(inverse*nv+v,minlength=len(ue)*nv).reshape(-1,nv)
        totals=hist.sum(1);counts.append(len(ue));singletons.append(float(np.sum(totals==1)/len(ue)))
        marginals.append(hist.sum(0)/hist.sum())
        reduced=hist//np.gcd.reduce(hist,axis=1)[:,None]
        unique,inv=np.unique(reduced,axis=0,return_inverse=True)
        local_weight=np.bincount(inv,weights=totals/len(v),minlength=len(unique))
        primitive.append(unique);weight.append(local_weight);owners.append(np.full(len(unique),a))
    if sparse_conditionals:
        inv=np.concatenate(primitive);nunique=len(signatures)
        conditional=torch.sparse_coo_tensor(torch.tensor([sparse_rows,sparse_cols],dtype=torch.long),
            torch.tensor(sparse_values,dtype=torch.double),(nunique,nv)).coalesce()
    else:
        unique,inv=np.unique(np.concatenate(primitive),axis=0,return_inverse=True)
        nunique=len(unique);conditional=torch.from_numpy(unique/unique.sum(1,keepdims=True))
    index=torch.from_numpy(np.stack([np.concatenate(owners),inv]))
    weights=torch.sparse_coo_tensor(index,torch.from_numpy(np.concatenate(weight)),(len(x),nunique)).coalesce()
    return EmpiricalBinaryMI(conditional,weights,torch.from_numpy(np.stack(marginals)),dict(
        visible_sites=len(visible),environment_sites=len(environment),states=states,buffer=buffer,
        unique_conditionals=nunique,sparse_conditionals=bool(sparse_conditionals),mean_observed_environment_states=float(np.mean(counts)),
        mean_fraction_environment_categories_seen_once=float(np.mean(singletons)),
        shuffle_visible_seed=shuffle_visible_seed,
        note='Exact empirical MI, not population MI. Unique/rare environments can cause severe finite-sample upward bias.'))


@dataclass
class LocalPairMI:
    groups: list
    diagnostics: dict

    def to(self, device, *, dtype=None, non_blocking=False):
        return LocalPairMI([group.to(device, dtype=dtype, non_blocking=non_blocking)
                            for group in self.groups], self.diagnostics)

    def information(self, table):
        return torch.stack([g.information(table) for g in self.groups]).mean(0)

    def prepare_selection(self):
        for group in self.groups:
            group.prepare_selection()
        return self

    def information_for(self, table, indices):
        return torch.stack([g.information_for(table, indices) for g in self.groups]).mean(0)


def categorical_local_pair_mi_statistics(x, *, flip=False, shuffle_visible_seed=None):
    """Average four I(H;two-site side patches), keeping the width-one buffer.

    This is deliberately NOT I(H;union of all patches); synergistic information
    can be lost. No physical labels or symmetrization of the kernel are used.
    """
    if flip or x.ndim!=5:raise ValueError('Requires 2D categorical fields without binary flip')
    offsets=[[[-2,0],[-2,1]],[[3,0],[3,1]],[[0,-2],[1,-2]],[[0,3],[1,3]]]
    groups=[lattice_mi_statistics(x,states=x.shape[-1],environment_offsets=o,
                                shuffle_visible_seed=shuffle_visible_seed) for o in offsets]
    return LocalPairMI(groups,dict(environment_mode='local_pairs',offsets=offsets,
        groups=[g.diagnostics for g in groups],aggregation='arithmetic mean of four separate MI values, not joint MI'))


def best_deterministic_binary_rule(stats,*,chunk=128):
    """Exhaustive search over all rules for up to four binary input sites."""
    nv=stats.marginal.shape[1]
    if nv>16:raise ValueError('Exhaustive binary rule search limited to four input sites')
    # Fix the label of the all-negative input to negative: complement rules
    # have exactly the same MI, reducing work by half without a physical prior.
    codes=torch.arange(0,2**nv,2,dtype=torch.long)
    powers=2**torch.arange(nv)
    mean_weight=torch.sparse.sum(stats.weights,dim=0).to_dense()/len(stats)
    scores=[]
    with torch.no_grad():
        for part in codes.split(chunk):
            rules=((part[:,None]//powers)%2).to(stats.marginal.dtype).T
            score=binary_entropy(stats.marginal@rules).mean(0)-mean_weight@binary_entropy(stats.conditional@rules)
            scores.append(score)
    score=torch.cat(scores);best=score.max()
    ties=codes[(score-best).abs()<1e-12].tolist();selected=min(ties)
    positive=((selected//powers)%2).to(stats.marginal.dtype)
    table=torch.stack((1-positive,positive),1)
    return table,dict(selected_code=selected,tied_codes_in_output_label_gauge=ties,
        exhaustive_rule_count=2**nv,evaluated_complement_representatives=len(codes),
        best_MI_nats=float(best),selected_P_plus=positive.tolist(),
        output_gauge='C(all-negative)=negative; output-complement rules have identical MI; no symmetry constraint during subsequent training')


def optimize_categorical_rule(stats,states,*,seed=91,starts=8,steps=600):
    """Finite-alphabet MI multistart search; not an exhaustive/global optimum."""
    generator=torch.Generator().manual_seed(seed)
    nv=stats.marginal.shape[1]
    logits=(.1*torch.randn(starts,nv,states,generator=generator,dtype=torch.double)).requires_grad_()
    optimizer=torch.optim.Adam([logits],lr=.05)
    mean_weight=torch.sparse.sum(stats.weights,dim=0).to_dense()/len(stats)
    def scores(tables):
        flat=tables.permute(1,0,2).reshape(nv,-1)
        return categorical_entropy((stats.marginal@flat).reshape(len(stats),-1,states)).mean(0)-mean_weight@categorical_entropy((stats.conditional@flat).reshape(len(stats.conditional),-1,states))
    best_score=-float('inf');best_table=None;history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        soft=torch.softmax(logits,dim=-1);score=scores(soft)
        (-score.sum()).backward();optimizer.step()
        if step==0 or (step+1)%50==0:
            with torch.no_grad():
                hard=torch.nn.functional.one_hot(logits.argmax(-1),states).double()
                values=scores(hard);value,index=values.max(0)
                if float(value)>best_score:best_score=float(value);best_table=hard[int(index)].clone()
            history.append(dict(step=step+1,best_hard_MI_nats=best_score))
    # Relabel outputs to match uniform input labels only when they already
    # form a bijection. This is a gauge, not a constraint on the search.
    width=round(np.log(nv)/np.log(states));uniform=torch.arange(states)*sum(states**i for i in range(width))
    labels=best_table[uniform].argmax(-1)
    bijective=len(torch.unique(labels))==states
    if bijective:best_table=best_table[:,labels]
    return best_table,dict(search='multistart soft categorical table + hard candidate selection',
        seed=seed,starts=starts,steps=steps,globally_optimal=False,best_MI_nats=best_score,
        selected_outputs=best_table.argmax(-1).tolist(),history=history,
        uniform_output_gauge_applied=bijective)
