"""Held-out microscopic nonlinear readout; no model optimization."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.q_feature_bottleneck import FeatureBottleneck,linear_probe
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('results/spin1_Qfeature_bottleneck_2400_seed91'))
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=args.run
    source=Path('results/spin1_Qonly_loss_candidates_exact_2400_seed91')
    loaded=load_run(source/'baseline.pt');tr,te=load_ensembles(loaded.config['data']['cache'])
    y=tr.couplings.numpy().reshape(len(tr.configurations),-1)[:,0];t=te.couplings.numpy().reshape(len(te.configurations),-1)[:,0]
    f=np.load(source/'feature_audit/baseline_features.npz');a=torch.from_numpy(f['pooled_local_5_train']);b=torch.from_numpy(f['pooled_local_5_test'])
    result={}
    with torch.no_grad():
        for path in sorted(root.glob('g*_s*.pt')):
            ck=torch.load(path,weights_only=True);net=FeatureBottleneck(ck['bottleneck_dim']);net.load_state_dict(ck['state'])
            aa=(a-ck['mean'])@ck['whitening'];bb=(b-ck['mean'])@ck['whitening'];ga=net(aa).numpy();gb=net(bb).numpy()
            # Scale by training spread only; k is fixed, never selected on the test data.
            sd=np.maximum(ga.std(0),1e-10);dist=np.sum(((gb[:,None]-ga[None])/sd)**2,axis=-1);ids=np.argsort(dist,axis=1)[:,:3]
            pred=y[ids].mean(1);mask=abs(t)>.2
            reconstruction=net.decoder(torch.from_numpy(gb)).numpy()
            decoded_K=linear_probe(aa.numpy(),reconstruction,y)
            result[path.stem]=dict(knn3_K_rmse=float(np.sqrt(np.mean((pred-t)**2))),knn3_K_r2=float(1-np.sum((pred-t)**2)/np.sum((t-t.mean())**2)),knn3_sign_accuracy=float(np.mean(np.sign(pred[mask])==np.sign(t[mask]))),reconstructed_feature_K_r2=float(1-np.sum((decoded_K-t)**2)/np.sum((t-t.mean())**2)))
    result['protocol']='Microscopic k=0 only; kNN3 Euclidean after train-only coordinate standardization; labels diagnostic only; original disjoint-K train/test split; one cache, small test set; nonlinear probe is not RG closure.'
    (root/'nonlinear_readout.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
