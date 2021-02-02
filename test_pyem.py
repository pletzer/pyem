import pytest
import pyem
import numpy as np

def test_read_table():
    data = pyem.read_table('data/case1_data.txt')
    assert data.shape[0] == 10
    assert data.shape[1] == 2


def test_read_trees():
    data = np.array(np.loadtxt('data/binary.txt'), np.float64)
    assert data.shape[0] == 39
    assert data.shape[1] == 13


def test_create_random_zig():
    n = 100
    G = 10
    z_ig = pyem.create_random_z_ig(n, G)
    print(z_ig)
    s_z = np.sum(z_ig, axis=1)
    assert np.all(s_z == 1)


def test_m_step():
    data = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]])
    z_ig = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    res = pyem.m_step(data, z_ig)
    assert len(res['pi_g']) == 2
    assert np.sum(res['pi_g']) == 1
    assert res['theta_gj'].shape[0] == 2
    assert res['theta_gj'].shape[1] == 3
    assert res['theta_gj'][0, 0] == 0
    assert res['theta_gj'][0, 1] == 1
    assert res['theta_gj'][0, 2] == 1
    assert res['theta_gj'][1, 0] == 1
    assert res['theta_gj'][1, 1] == 0
    assert res['theta_gj'][1, 2] == 0


def test_e_step_bernoulli():

    data = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]])
    z_ig = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    res = pyem.m_step(data, z_ig)
    p_ig = pyem.e_step_bernoulli(data, res['pi_g'], res['theta_gj'])

    print(p_ig)

    assert p_ig[0, 0] == 1
    assert p_ig[0, 1] == 0

    assert p_ig[1, 0] == 1
    assert p_ig[1, 1] == 0

    assert p_ig[2, 0] == 0
    assert p_ig[2, 1] == 1

    assert p_ig[3, 0] == 0
    assert p_ig[3, 1] == 1


def test_cluster_bernoulli():
    data = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]) 
    res = pyem.cluster_bernoulli(data, G=2, maxiter=10, max_diff=1.e-6, seed=123)
    assert abs(res['diff']) < 1.e-6
    assert abs(res['theta_gj'][0, 0] - (1.)) < 1.e-6
    assert abs(res['theta_gj'][0, 1] - (0.)) < 1.e-6
    assert abs(res['theta_gj'][0, 1] - (0.)) < 1.e-6
    assert abs(res['theta_gj'][1, 0] - (0.)) < 1.e-6
    assert abs(res['theta_gj'][1, 1] - (1.)) < 1.e-6
    assert abs(res['theta_gj'][1, 1] - (1.)) < 1.e-6


