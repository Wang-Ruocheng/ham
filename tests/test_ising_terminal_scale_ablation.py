"""No training/sampling tests: matched config, full DR, response and guards."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy

import numpy as np
import torch

from scripts.run_scalar_lr_sweep import make_configs
from scripts.run_ising2d_terminal_scale_ablation import matched_config, differences, DATA_SHA, gpu_snapshot, run
from scripts.report_ising_local_response import predict_and_derivative, polynomial_response, root_closure_conflicts


class TerminalScaleTests(unittest.TestCase):
    def manifest(self):
        cfg=make_configs(Path('/unused/L108.pt'),epochs=12000,seed=91,threads=2)['multistep']
        cfg['training']['max_rg_steps']=cfg['evaluation']['max_rg_steps']=3
        return dict(config=cfg,data_sha256=DATA_SHA)

    def test_only_depth_changes_and_source_not_mutated(self):
        source=self.manifest(); before=deepcopy(source)
        result=matched_config(source)
        self.assertEqual(differences(result,source['config']),['evaluation.max_rg_steps','training.max_rg_steps'])
        self.assertEqual(source,before)
        self.assertEqual(result['training']['epochs'],12000)

    def test_wrong_source_rejected(self):
        source=self.manifest();source['data_sha256']='0'*64
        with self.assertRaises(ValueError):matched_config(source)

    def test_full_derivative_includes_identity(self):
        class Residual:
            def advance_theory(self,x):return x+2*x+0.1
        x=np.array([-.1,.1])
        y,d=predict_and_derivative(Residual(),x)
        np.testing.assert_allclose(y,3*x+.1)
        np.testing.assert_allclose(d,3.)

    def test_exact_response_and_paired_uncertainty(self):
        k=np.linspace(-.02,.02,9)
        means=np.stack([k,2*k,4*k],1)
        boot=np.repeat(means[None],40,axis=0)
        report=polynomial_response(k,means,boot,2*means[:,:-1],2*boot[:,:,:-1],0.,.02)
        for row in report['transitions']:
            self.assertAlmostEqual(row['empirical_response'],2.)
            self.assertAlmostEqual(row['response_mismatch'],0.)

    def test_conflict_detected_and_unsupported_root_retained(self):
        k=np.linspace(-1.,1.,9)
        means=np.stack([k,2*k,4*k+.3],1)
        boot=np.repeat(means[None],30,0)
        results=root_closure_conflicts(k,means,boot,[dict(g=[0.]),dict(g=[9.])])
        self.assertAlmostEqual(results[0]['cross_scale_differences'][0]['next_value_difference'],-.3)
        self.assertEqual(results[1]['branches'],[])

    def test_selected_gpu_index_queried(self):
        with patch('scripts.run_ising2d_terminal_scale_ablation.subprocess.check_output',side_effect=['GPU-id, NVIDIA GeForce RTX 3090, 24000\n','']) as call:
            result=gpu_snapshot(1)
        self.assertEqual(result['index'],1)
        self.assertEqual(result['compute_pids'],[])
        for args in call.call_args_list:self.assertEqual(args.args[0][2],'1')

    def test_refuse_existing_output(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):run(Path('/unused'),Path(d),1)


if __name__=='__main__':unittest.main()
