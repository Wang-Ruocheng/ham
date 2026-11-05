"""小型合成构型执行检查，不是物理采样或科学结果。"""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parent))
import collect
from recurrent_rg.checkpoints import build_model, isolated_rng
from recurrent_rg.config import ModelConfig


class CollectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        gen=torch.Generator().manual_seed(341)
        cls.spins=(2*torch.randint(0,2,(32,12,12),generator=gen)-1).double()

    def test_plaquette_order(self):
        actual=collect.frequencies2(self.spins)
        expected=collect.plaquette_distribution(self.spins[:,None,:,:,None])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_q_histogram_matches_original_encoder(self):
        cfg=ModelConfig(dimension=2,theory_dim=1,hidden=96,width=3,block_factor=2,
                        binary_lookup=True,affine_reconstruction=True)
        with isolated_rng('cpu',91):model=build_model(cfg,2).double().eval()
        for param in model.parameters():param.requires_grad_(False)
        table=collect.q_feature_table(model)
        frequency=collect.frequencies3(self.spins,table)
        with torch.no_grad():
            a=model.theory_encoder.head((frequency@table['values']).mean(0))
            b=model.theory_encoder(self.spins[None,:,:,:,None])[0]
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_exact_motif_statistics_reuse(self):
        class Table:pass
        from recurrent_rg.binary_pooling import binary_pattern_table
        powers,_=binary_pattern_table(9,self.spins)
        table={'powers':powers,'offsets':tuple((a,b) for a in (-1,0,1) for b in (-1,0,1))}
        _,p2,score=collect.statistics_from_motifs(self.spins,table)
        torch.testing.assert_close(p2,collect.frequencies2(self.spins),rtol=0,atol=0)
        torch.testing.assert_close(score,collect.operator_values(self.spins)[:,:6],rtol=0,atol=0)

    def test_longrange_symmetries_and_uniform(self):
        a=collect.longrange(self.spins)
        torch.testing.assert_close(a,collect.longrange(-self.spins),rtol=1e-11,atol=1e-13)
        torch.testing.assert_close(a,collect.longrange(torch.roll(self.spins,(2,3),(-2,-1))),rtol=1e-11,atol=1e-13)
        uniform=collect.longrange(torch.ones_like(self.spins))
        torch.testing.assert_close(uniform[:,:2],torch.ones_like(uniform[:,:2]))
        self.assertLess(float(uniform[:,2:].abs().max()),1e-25)


if __name__=='__main__':unittest.main()
