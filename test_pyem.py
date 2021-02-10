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


def test_m_step_row():
    data = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]])
    z_ig = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    res = pyem.m_step_row(data, z_ig)
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


def test_e_step_row_bernoulli():

    data = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]])
    z_ig = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    res = pyem.m_step_row(data, z_ig)
    p_ig = pyem.e_step_row_bernoulli(data, res['pi_g'], res['theta_gj'])

    print(p_ig)

    assert p_ig[0, 0] == 1
    assert p_ig[0, 1] == 0

    assert p_ig[1, 0] == 1
    assert p_ig[1, 1] == 0

    assert p_ig[2, 0] == 0
    assert p_ig[2, 1] == 1

    assert p_ig[3, 0] == 0
    assert p_ig[3, 1] == 1


def test_cluster_row_bernoulli():
    data = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]) 
    res = pyem.cluster_row_bernoulli(data, G=2, maxiter=10, max_diff=1.e-6, seed=123)
    assert abs(res['diff']) < 1.e-6
    assert abs(res['theta_gj'][0, 0] - (1.)) < 1.e-6
    assert abs(res['theta_gj'][0, 1] - (0.)) < 1.e-6
    assert abs(res['theta_gj'][0, 1] - (0.)) < 1.e-6
    assert abs(res['theta_gj'][1, 0] - (0.)) < 1.e-6
    assert abs(res['theta_gj'][1, 1] - (1.)) < 1.e-6
    assert abs(res['theta_gj'][1, 1] - (1.)) < 1.e-6


def test_e_step_bi_bernoulli():

    pi_r = [3./27, 11./27]
    kappa_c = [3./18., 5./18., 7./18.]
    theta_rc = [[0., 0.1, 0.2],
                [0.3, 0.4, 0.5]]

    R = len(pi_r)
    C = len(kappa_c)

    data = [[0., 0., 0.,  1., 0., 0.,  1., 1., 0.], # r = 1
            [0., 0., 0.,  0., 0., 0.,  0., 0., 0.], # r = 1
            [0., 0., 0.,  0., 0., 0.,  0., 0., 0.], # r = 1
            [1., 1., 1.,  1., 1., 1.,  1., 1., 1.], # r = 2
            [0., 0., 0.,  1., 0., 0.,  1., 1., 0.], # r = 2
            [0., 0., 0.,  0., 0., 0.,  0., 0., 0.]] # r = 2
    data = np.array(data)
    n, p = data.shape

    resE = pyem.e_step_bi_bernoulli(data, np.array(pi_r), np.array(kappa_c), np.array(theta_rc))
    print(resE)

    assert abs(np.sum(np.sum(resE['z_ir'], axis=1) - 1.)) < 1.e-10
    assert abs(np.sum(np.sum(resE['x_jc'], axis=1) - 1.)) < 1.e-10


def test_cluster_bi_bernoulli_2x1():

    R = 2
    C = 1
    n = 10
    p = 4

    data = np.zeros((n, p), np.float64)
    data[n//2:, :] = 1

    pi_r = np.array([0.5, 0.5])
    kappa_c = np.array([1.])
    theta_rc = np.array([[0.0],
                         [1.0]])

    resE = pyem.e_step_bi_bernoulli(data, pi_r, kappa_c, theta_rc)
    res = pyem.m_step_bi(data, resE['z_ir'], resE['x_jc'])
    #res = pyem.cluster_bi_bernoulli_em(data, R=R, C=C, maxiter=100, max_diff=1.e-10, seed=123)

    print(res)

    assert res['pi_r'][0] == 0.5
    assert res['pi_r'][1] == 0.5
    assert res['kappa_c'][0] == 1.
    assert res['theta_rc'][0,0] == 0
    assert res['theta_rc'][1,0] == 1


def test_cluster_bi_bernoulli_2x2():

    R = 2
    C = 2

    data = np.array([[ 0., 0.,  1., 1.],
                     [ 0., 0.,  1., 1.], 
                     [ 1., 1.,  0.,  0.],
                     [ 1., 1.,  0.,  0.]])


    pi_r = np.array([0.5, 0.5])
    kappa_c = np.array([0.5, 0.5])
    theta_rc = np.array([[0.0, 1.0],
                         [1.0, 0.0]])

    resE = pyem.e_step_bi_bernoulli(data, pi_r, kappa_c, theta_rc)

    print(f"z_ir = {resE['z_ir']}")
    print(f"x_jc = {resE['x_jc']}")

    res = pyem.m_step_bi(data, resE['z_ir'], resE['x_jc'])

    print(res)

    res = pyem.cluster_bi_bernoulli_em(data, R=R, C=C, maxiter=100, max_diff=1.e-10, seed=123)

    print(res)

    assert abs(res['pi_r'][0] - 0.5) < 1.e-10
    assert abs(res['pi_r'][1] - 0.5) < 1.e-10
    assert abs(res['kappa_c'][0] - 0.5) < 1.e-10
    assert abs(res['kappa_c'][1] - 0.5) < 1.e-10
    assert abs(res['theta_rc'][0,0] - 0.5) < 1.e-10
    assert abs(res['theta_rc'][0,1] - 0.5) < 1.e-10
    assert abs(res['theta_rc'][1,0] - 0.5) < 1.e-10
    assert abs(res['theta_rc'][1,1] - 0.5) < 1.e-10



def test_m_step_bi():

    pi_r = [3./27, 11./27]
    kappa_c = [3./18., 5./18., 7./18.]
    theta_rc = [[0., 0.1, 0.2],
                [0.3, 0.4, 0.5]]

    R = len(pi_r)
    C = len(kappa_c)

    data = [[0., 0., 0.,  1., 0., 0.,  1., 1., 0.], # r = 1
            [0., 0., 0.,  0., 0., 0.,  0., 0., 0.], # r = 1
            [0., 0., 0.,  0., 0., 0.,  0., 0., 0.], # r = 1
            [1., 1., 1.,  1., 1., 1.,  1., 1., 1.], # r = 2
            [0., 0., 0.,  1., 0., 0.,  1., 1., 0.], # r = 2
            [0., 0., 0.,  0., 0., 0.,  0., 0., 0.]] # r = 2
    data = np.array(data)
    n, p = data.shape

    pi_r = np.array(pi_r)
    kappa_c = np.array(kappa_c)
    theta_rc = np.array(theta_rc)

    for iteration in range(10):
        resE = pyem.e_step_bi_bernoulli(data, pi_r, kappa_c, theta_rc)
        resM = pyem.m_step_bi(data, resE['z_ir'], resE['x_jc'])
        pi_r = resM['pi_r']
        kappa_c = resM['kappa_c']
        theta_rc = resM['theta_rc']
        print(f'iteration = {iteration} theta_rc = {theta_rc}')

    assert np.max(theta_rc) <= 1.
    assert np.max(theta_rc) >= 0.


def test_cluster_bi_bernoulli_trees_em():

    R = 2
    C = 2

    data = np.array(np.loadtxt('data/binary.txt'), np.float64)
    assert data.shape[0] == 39
    assert data.shape[1] == 13

    res = pyem.cluster_bi_bernoulli_em(data, R=R, C=C, maxiter=1000, max_diff=1.e-10, seed=123)

    print(res)
    assert False


def test_cluster_bi_bernoulli_trees_me():

    R = 2
    C = 2

    data = np.array(np.loadtxt('data/binary.txt'), np.float64)
    assert data.shape[0] == 39
    assert data.shape[1] == 13

    res = pyem.cluster_bi_bernoulli_me(data, R=R, C=C, maxiter=1000, max_diff=1.e-10, seed=123)

    print(res)
    assert False


def xtest_cluster_bi_bernoulli():

    pi_r = [3./27, 11./27]
    kappa_c = [3./18., 5./18., 7./18.]
    theta_rc = [[0., 0.1, 0.2],
                [0.3, 0.4, 0.5]]

    data = [[0., 0., 0.,  1., 0., 0.,  1., 1., 0.], # r = 1
            [0., 0., 0.,  0., 0., 0.,  0., 0., 0.], # r = 1
            [0., 0., 0.,  0., 0., 0.,  0., 0., 0.], # r = 1
            [1., 1., 1.,  1., 1., 1.,  1., 1., 1.], # r = 2
            [0., 0., 0.,  1., 0., 0.,  1., 1., 0.], # r = 2
            [0., 0., 0.,  0., 0., 0.,  0., 0., 0.]] # r = 2
    data = np.array(data)


    #resR = pyem.cluster_row_bernoulli(data, G=2, maxiter=100, max_diff=1.e-6, seed=123)
    resB = pyem.cluster_bi_bernoulli_em(data, R=2, C=3, maxiter=100, max_diff=1.e-6, seed=123)

    print(resB)

    assert False
    

