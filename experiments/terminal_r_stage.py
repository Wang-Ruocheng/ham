"""Terminal stage for the standard finite-alphabet QCR driver."""
import copy
import json
from pathlib import Path
import time

import torch

from recurrent_rg.checkpoints import checkpoint_payload, qcr_state_dict
from recurrent_rg.r_refinement import component_optimizer_steps
from recurrent_rg.terminal_r_fit import collect_train_targets, fit_r, freeze_qc


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run_terminal_stage(model, optimizer, train_configurations, config, output,
                       metadata, generator, joint_epoch, *, budget_check=None):
    """Preserve root joint artifacts; publish a separate final checkpoint pointer."""
    root = Path(output); out = root/'r_only'; out.mkdir(exist_ok=False)
    pipeline = dict(schema='qcr_joint_then_terminal_r_v1', status='running', workflow_complete=False,
        joint_updates=joint_epoch, joint_checkpoint='checkpoint_last.pt',
        final_checkpoint='r_only/checkpoint_last.pt', current='fixed_TRAIN_cache', error=None)
    write_json(root/'pipeline.json', pipeline)
    started = time.monotonic()
    try:
        if budget_check: budget_check()
        freeze_qc(model, model.theory_rg)

        @torch.no_grad()
        def encode(batch):
            if config.data.spin_flip_augment:
                a, b = batch.chunk(2, dim=1); batch = torch.cat((a, -a, b, -b), dim=1)
            half = batch.shape[1]//2
            return torch.stack([.5*(model.theory_encoder(x[:, :half])+model.theory_encoder(x[:, half:]))
                                for x in model.coarse_grain(batch)], 1)

        targets, cache = collect_train_targets(model, train_configurations, encode,
            samples=config.training.sample_batch, systems=config.training.system_batch,
            repeats=config.training.terminal_r_cache_repeats, budget_check=budget_check)
        torch.save(targets, out/'train_targets.pt'); write_json(out/'cache.json', cache)
        stage_metadata = copy.deepcopy(metadata)
        stage_metadata['joint_R_optimization'] = stage_metadata.get('R_optimization', {})
        stage_metadata['R_optimization'] = dict(separate_R_only_stage=True,
            sampling='Fixed TRAIN cache; one full-cache mean per additional Adam update',
            learning_rate='Original R Adam moments and current LR retained',
            loss_changed=False, labels_used=False, Q_C_updates=0)
        stage_metadata.update(training_driver='terminal_fixed_QC_R',
            terminal_R_stage=True, terminal_R_fit_completed=False, joint_updates=joint_epoch,
            optimized_modules=['R'], frozen_modules=['Q', 'C'], labels_used_in_training=False,
            target_policy='Fixed TRAIN two-half-mean encodings; cached subset/C draws, full-cache mean',
            selection='TRAIN plateau or update cap; final endpoint, never TEST/root-selected')
        write_json(out/'manifest.json', stage_metadata)

        def save(row, rows):
            write_json(out/'history.json', rows)
            payload = checkpoint_payload(model, optimizer, generator, joint_epoch,
                metadata=stage_metadata, history=[])
            payload.update(extra_R_updates=row['extra_R_updates'],
                optimizer_steps=component_optimizer_steps(model, optimizer), terminal_R_history=rows)
            torch.save(payload, out/'checkpoint_last.pt')
            pipeline.update(current='R_only_training', extra_R_updates=row['extra_R_updates'], train_loss=row['train_loss'])
            write_json(root/'pipeline.json', pipeline)
            print(f"terminal R +{row['extra_R_updates']}: fixed TRAIN loss={row['train_loss']:.8g}", flush=True)

        result = fit_r(model, model.theory_rg, {'Q':model.theory_encoder, 'C':model.rg_kernel},
            optimizer, targets, config.loss, max_steps=config.training.terminal_r_max_steps,
            callback=save, budget_check=budget_check)
        stage_metadata['terminal_R_fit_completed']=True
        write_json(out/'manifest.json', stage_metadata)
        save(result['history'][-1],result['history'])
        torch.save(qcr_state_dict(model), out/'qcr_model.pt')
        result.update(joint_updates=joint_epoch, optimizer_steps=component_optimizer_steps(model, optimizer),
            total_stage_seconds=time.monotonic()-started, cache=cache)
        write_json(out/'summary.json', result)
        from scripts.report_phi4_hopping_qcr import _setup_plot
        plt=_setup_plot(None); fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        ax.semilogy([r['extra_R_updates'] for r in result['history']],
                    [r['train_loss'] for r in result['history']],'o-')
        ax.set(xlabel='额外 R 更新次数',ylabel='固定 TRAIN 缓存闭环损失',title='末尾固定 Q/C 的 R-only 拟合')
        ax.grid(alpha=.2);fig.savefig(out/'loss_curves.png',dpi=160);plt.close(fig)
        pipeline.update(status='trained', workflow_complete=False, model_complete=True, current='final_report',
            stop_reason=result['stop_reason'], extra_R_updates=result['actual_extra_R_updates'])
        write_json(root/'pipeline.json', pipeline)
        return result
    except Exception as exc:
        pipeline.update(status='incomplete', workflow_complete=False, error=str(exc))
        write_json(root/'pipeline.json', pipeline)
        raise
