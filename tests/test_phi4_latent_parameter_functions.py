import numpy as np
import pytest
from scripts.plot_phi4_latent_parameter_functions import group_statistics, correlation, edges


def test_grid_orientation_chain_mean_sem_and_conditional_correlations():
    params = np.array([(r,u,0) for r in (-2.,-1.,0.) for u in (1.,2.)])
    labels = params.repeat(4,axis=0)
    cells = np.arange(6).repeat(4)
    base = np.stack((labels[:,0]+2*labels[:,1],labels[:,0]**2),-1)
    g = np.repeat(base[:,None,:],3,axis=1)
    g += np.tile([-1.,-1.,1.,1.],6)[:,None,None]*.02
    result=group_statistics(g,labels,cells)
    assert result['grid'].shape==(2,3,3,2)
    assert np.isclose(result['grid'][1,0,0,0],2.)
    assert np.isclose(result['sem'][0,0,0],.02/np.sqrt(3))
    assert np.isclose(result['statistics'][0]['components'][0]['fixed_u'][0]['pearson'],1.)
    assert np.allclose(edges(np.array([1.,2.,3.])),[.5,1.5,2.5,3.5])


def test_zero_linear_correlation_does_not_mean_no_dependence():
    x=np.linspace(-1,1,9)
    assert abs(correlation(x,x**2)['pearson'])<1e-14
    assert correlation(np.ones(9),x)['pearson'] is None


def test_reject_missing_grid_cell():
    labels=np.array([[-1,1,0],[-1,2,0],[0,1,0]]).repeat(4,axis=0)
    with pytest.raises(ValueError):group_statistics(np.zeros((12,3,2)),labels,np.arange(3).repeat(4))
