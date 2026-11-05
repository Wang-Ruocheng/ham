"""Same frozen evaluator, with the new integration provenance kept explicit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from experiments import phi4_ru_followup_evaluate as frozen
from experiments.phi4_qcr import sha256_file,write_json


def verify_fine_plan(source_root,data_root,validation):
    plan=json.loads((Path(data_root)/"plan.json").read_text())
    if (Path(plan["source_root"]).resolve()!=Path(source_root).resolve()
            or Path(plan["validation"]).resolve()!=Path(validation).resolve()):
        raise ValueError("Fine evaluation source or selected R provenance differs from parent plan")
    if len(plan["subsets"])!=13:
        raise ValueError("Fine evaluator requires all13 predeclared child datasets")
    for row in plan["subsets"]:
        options=row["physical"]["options"]
        expected=(.04,24) if row["size"]==108 else (.08,12)
        if (options["step_size"],options["leapfrog_steps"])!=expected:
            raise ValueError("New physical data integration differs from the fixed schedule")
    return plan


def run(source_root,data_root,validation,output):
    plan=verify_fine_plan(source_root,data_root,validation)
    code_sha=sha256_file(__file__)
    frozen.run(source_root,data_root,validation,output)
    output=Path(output)
    try:
        if sha256_file(__file__)!=code_sha:
            raise RuntimeError("Fine evaluator wrapper changed during execution")
        write_json(output/"fine_integration_receipt.json",dict(schema_version=5,
            parent_plan_sha256=sha256_file(Path(data_root)/"plan.json"),
            wrapper_source_sha256=code_sha, evaluator_source_sha256=sha256_file(frozen.__file__),
            integration_by_size={"36":dict(step_size=.08,leapfrog_steps=12),
                                 "108":dict(step_size=.04,leapfrog_steps=24)},
            production_parameters_adapted=False, Q_updates=0,C_updates=0,R_updates=0,
            physical_seed_relationship="同seed的积分修正版本；原失败链不并入本批或增加独立样本数",
            source_root=plan["source_root"],validation=plan["validation"]))
        write_json(output/"public_files.json",{p.name:sha256_file(p) for p in output.iterdir()
                   if p.is_file() and p.name!="public_files.json"})
    except BaseException as error:
        write_json(output/"status.json",dict(status="failed",workflow_complete=False,returncode=1,
            error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc(),
            inherited_evaluation_completed=True))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source_root","data_root","validation","output"):
        parser.add_argument("--"+name.replace("_","-"),required=True)
    run(**vars(parser.parse_args()))
