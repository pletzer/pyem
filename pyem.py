import numpy as np


def read_table(filename):
    data = []
    with open(filename) as f:
        # header
        f.readline()
        for line in f.readlines():
            data.append([float(v) for v in line.split(' ')[1:]])
    return np.array(data)


def create_random_z_ig(n, G):
    z_ig = np.zeros((n, G), np.float64)
    for i in range(n):
        z_ig[ i, np.random.choice(G) ] = 1
    return z_ig



def likelihood_bernoulli(data, z_ig, pi_g, theta_gj):

    G = pi_g.shape[0]
    n = data.shape[0]

    # G x n x P
    th = np.swapaxes(th, axis1=1, axis2=0)

    # G x n x P
    x = np.tile(data, (G, 1, 1))

    # G x n
    f_gi = np.prod(th**x * (1. - th)**(1 - x), axis=2)

    res = 0.
    for i in range(n):
        res2 = 0.
        for g in range(G):
            res2 += z_ig[i, g] * pi_g[g] * f_gi[g, i]
        res += np.log(res2)

    return res



def m_step(data, z_ig):
    n = data.shape[0]
    piHat_g = np.sum(data, axis=0)/n
    thetaHat_gj = z_ig.transpose().dot(data)/np.sum(z_ig, axis=0)
    return {'pi_g': piHat_g, 'theta_gj': thetaHat_gj}


def e_step_bernoulli(data, pi_g, theta_gj):
    
    n = data.shape[0]
    G = len(pi_g)

    # array of size n x G x P
    th = np.tile(theta_gj, (n, 1, 1))

    # G x n x P
    th = np.swapaxes(th, axis1=1, axis2=0)

    # G x n x P
    x = np.tile(data, (G, 1, 1))

    # G x n
    f_gi = np.prod(th**x * (1. - th)**(1 - x), axis=2)

    # n
    sum_pi_f = p_ig.dot(f_gi)

    p_ig = np.zeros((n, G), np.float64)
    for g in range(G):
        for i in range(n):
            p_ig[i, g] = pi_g[g] * f_ig[i, g] / sum_pi_f[i]


    return p_ig



def cluster_bernoulli(data, G=2, maxiter=100, max_diff=1.e-6, seed=123):

    np.random.seed(seed)

    n = data.shape[0]

    # random initial guess 
    z_ig = np.zeros((n, G), np.float64)

    # randomly assign 1 for each row
    for i in range(n):
        z_ig[i, np.random.randint(G)]

    diff = float('inf')
    iteration = 0
    while iteration < maxiter and diff > max_diff:

        z_old = z_ig.copy()

        res = m_step(data, z_ig)

        p_ig = e_step_bernoulli(data, res['p_ig'], res['theta_gj'])

        # sample z_ig from p_ig
        z_ig = p_ig

        iteration += 1

    if iteration >= maxiter:
        print(f'Warning: reached max number of iterations! diff = {diff}')


    return res


