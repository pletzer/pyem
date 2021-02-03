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
    p = data.shape[1]
    G = z_ig.shape[1]

    piHat_g = np.sum(z_ig, axis=0)/n

    thetaHat_gj = np.zeros((G, p), np.float64)
    for g in range(G):
        for j in range(p):
            numer = 0
            denom = 0
            for i in range(n):
                numer += z_ig[i, g]*data[i, j]
                denom += z_ig[i, g]
            thetaHat_gj[g, j] = numer / denom

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
    sum_pi_f = pi_g.dot(f_gi)

    p_ig = np.zeros((n, G), np.float64)
    for g in range(G):
        for i in range(n):
            p_ig[i, g] = pi_g[g] * f_gi[g, i] / sum_pi_f[i]


    return p_ig



def cluster_bernoulli(data, G=2, maxiter=100, max_diff=1.e-6, seed=123):

    np.random.seed(seed)

    n = data.shape[0]

    # randomly assign 1 for each row
    z_ig = create_random_z_ig(n, G)

    diff = float('inf')
    iteration = 0
    while iteration < maxiter and diff > max_diff:

        z_old = z_ig.copy()

        res = m_step(data, z_ig)

        res['p_ig'] = e_step_bernoulli(data, res['pi_g'], res['theta_gj'])

        # sample z_ig from p_ig
        z_ig = res['p_ig']
        diff = np.sum(abs(z_ig - z_old))

        iteration += 1

    if iteration >= maxiter:
        print(f'Warning: reached max number of iterations {maxiter}! diff = {diff}')

    res['iterations'] = iteration
    res['diff'] = diff

    return res


