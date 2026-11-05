"""Plan or execute the separately authorized fine-integration phi4 child."""
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
              "NUMBA_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_name]="1"

import argparse
import json
from pathlib import Path

from recurrent_rg.data.phi4_ru_fine_pipeline import Phi4FineOptions,generation_plan,generate_dataset


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--execute",action="store_true")
    for name in ("sizes","r_values","u_values"):
        parser.add_argument("--"+name.replace("_","-"),nargs="+",type=int if name=="sizes" else float,
                            default=list(getattr(Phi4FineOptions,name)))
    for name in ("h","step_size","cluster_probability"):
        parser.add_argument("--"+name.replace("_","-"),type=float,default=getattr(Phi4FineOptions,name))
    for name in ("dimension","chains","samples","burn_in","spacing","leapfrog_steps","seed",
                 "test_every","test_offset","max_trajectories","max_force_site_evaluations",
                 "max_cluster_updates","max_cluster_neighbor_visits","max_output_bytes","max_memory_bytes"):
        parser.add_argument("--"+name.replace("_","-"),type=int,default=getattr(Phi4FineOptions,name))
    values=vars(parser.parse_args(argv))
    output,execute=values.pop("output"),values.pop("execute")
    for name in ("sizes","r_values","u_values"):
        values[name]=tuple(values[name])
    options=Phi4FineOptions(**values)
    plan=generation_plan(options)
    if not execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2,allow_nan=False))
        return plan
    if output is None:
        parser.error("--execute requires --output")
    result=generate_dataset(output,options)
    print(json.dumps(dict(status="complete",output=str(output.resolve()),conditions=len(result["shards"]),
                          warning_count=result["warning_count"],training_updates=0,equilibrium_certified=False)))
    return result


if __name__=="__main__":
    main()
