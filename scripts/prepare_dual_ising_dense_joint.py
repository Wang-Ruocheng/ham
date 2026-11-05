"""Merge existing training samples only; leave original and critical tests untouched."""
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from recurrent_rg.data.ensembles import load_ensembles,EnsembleDataset


def main():
    source=Path('results/dual_ising2d_g2_uniformK1K2_fixed_1200_seed91')
    tr,te=load_ensembles('data_cache/dual_ising2d_uniformK1K2_L16_n128_seed91/ensembles.pt')
    dense=np.load(source/'followup_four_directions/dense_samples.npz')
    states=dense['states'];xs=torch.nn.functional.one_hot(torch.from_numpy(states.astype(np.int64)).reshape(-1,128,16,16),4).to(torch.uint8)
    ys=torch.from_numpy(np.repeat(dense['couplings'],states.shape[1],axis=0)).double()
    combined=EnsembleDataset(torch.cat([tr.configurations,xs]),torch.cat([tr.couplings,ys]))
    combined.validate();te.validate()
    assert not set(map(tuple,combined.couplings.tolist())) & set(map(tuple,te.couplings.tolist()))
    audit=np.load(source/'critical_sampling_audit/trajectories.npz')['couplings']
    assert not np.isclose(combined.couplings.numpy()[:,None,:],audit[None,:,:]).all(-1).any()
    out=Path('data_cache/dual_ising2d_uniform_plus_dense_L16_seed91');out.mkdir(exist_ok=False)
    torch.save(dict(schema_version=1,train=asdict(combined),test=asdict(te)),out/'ensembles.pt')
    load_ensembles(out/'ensembles.pt')
    meta=dict(train_ensembles=len(combined.configurations),test_ensembles=len(te.configurations),samples=128,L=16,
              train_cells=len(torch.unique(combined.couplings,dim=0)),test_cells=len(torch.unique(te.couplings,dim=0)),
              original_train_ensembles=len(tr.configurations),added_ensembles=len(xs),
              note='Original grid heldout split and independent exact-critical audit untouched. Coupling labels used only for leakage checks and reporting.')
    (out/'sampling.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)


if __name__=='__main__':main()
