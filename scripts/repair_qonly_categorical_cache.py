"""Replay fixed C draws; canonicalize hard states before integer serialization."""
import json
from pathlib import Path
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.runner import file_fingerprint

torch.set_num_threads(2)
old=Path('results/spin1_Qonly_separation_sweep_2400_seed91')
out=Path('results/spin1_Qonly_cache_exact_seed91');out.mkdir(exist_ok=False)
m=json.loads((old/'manifest.json').read_text());loaded=load_run(m['source'])
tr,te=load_ensembles(m['config']['data']['cache']);torch.manual_seed(9191);result={};checks={}
with torch.no_grad():
    for split,ds in [('train',tr),('test',te)]:
        rows=[];maxerr=0.
        for x in ds.configurations:
            levels=loaded.model.coarse_grain(x[None].double())
            hard=[torch.nn.functional.one_hot(v.argmax(-1),3).to(torch.uint8) for v in levels]
            maxerr=max(maxerr,max(float(abs(v-h.double()).max()) for v,h in zip(levels,hard)))
            assert maxerr<1e-10
            rows.append(hard)
        result[split]=[torch.cat([r[k] for r in rows]) for k in range(3)]
        assert all(torch.all(v.sum(-1)==1) for v in result[split])
        checks[split]=dict(max_float_roundoff=maxerr,invalid_sites=0)
torch.save(result,out/'frozen_levels.pt')
m.update(cached_levels_sha256=file_fingerprint(out/'frozen_levels.pt'),reference_run=str(old.resolve()),serialization='argmax then one_hot, never direct floating-point-to-integer truncation',validation=checks)
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
invalid=Path('results/spin1_Qonly_loss_candidates_2400_seed91')
(invalid/'INVALID_CACHE_DIAGNOSTIC.json').write_text(json.dumps(dict(valid_for_comparison=False,reason='Coarse one-hot float values directly cast to uint8; ~7-8% invalid coarse sites. Do not use candidate results.',replacement='results/spin1_Qonly_loss_candidates_exact_2400_seed91'),indent=2)+'\n')
print(json.dumps(checks))
