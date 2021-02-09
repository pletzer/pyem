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



def likelihood_row_bernoulli(data, z_ig, pi_g, theta_gj):

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



def m_step_row(data, z_ig):
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



def e_step_row_bernoulli(data, pi_g, theta_gj):
    
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



def cluster_row_bernoulli(data, G=2, maxiter=100, max_diff=1.e-6, seed=123):

    np.random.seed(seed)

    n = data.shape[0]

    # randomly assign 1 for each row
    z_ig = create_random_z_ig(n, G)

    diff = float('inf')
    iteration = 0
    while iteration < maxiter and diff > max_diff:

        z_old = z_ig.copy()

        res = m_step_row(data, z_ig)

        res['p_ig'] = e_step_row_bernoulli(data, res['pi_g'], res['theta_gj'])

        # sample z_ig from p_ig
        z_ig = res['p_ig']
        diff = np.sum(abs(z_ig - z_old))

        iteration += 1

    if iteration >= maxiter:
        print(f'Warning: reached max number of iterations {maxiter}! diff = {diff}')

    res['iterations'] = iteration
    res['diff'] = diff

    return res


def m_step_bi(data, z_ir, x_jc):
    n = data.shape[0]
    p = data.shape[1]
    R = z_ir.shape[1]
    C = x_jc.shape[1]

    # Appendix A.1 of Multivariate methods using mixtures, correspondance analysis, scaling and patter-detection
    piHat_r = np.sum(z_ir, axis=0)/n
    kappaHat_c = np.sum(x_jc, axis=0)/p

    thetaHat_rc = z_ir.transpose().dot(data.dot(x_jc))
    thetaHat_rc /= n*p * piHat_r.reshape((R,1)).dot(kappa_c.reshape(1,C))

    return {'pi_r': piHat_r, 'kappa_c': kappaHat_c, 'theta_rc': thetaHat_rc}



def e_step_bi_bernoulli(data, pi_r, kappa_c, theta_rc):
    
    n = data.shape[0]
    p = data.shape[1]
    R = len(pi_r)
    C = len(kappa_c)


    g_cj = np.zeros((C, p), np.float64)
    zHat_ir = np.zeros((n, R), np.float64)
    for i in range(n):

        sum_r = 0.

        for r in range(R):

            for c in range(C):
                for j in range(p):
                    g_cj[c, j] = theta_rc[r, c]**data[i, j] * (1. - theta_rc[r, c])**(1. - data[i, j])

            zHat_ir[i, r] = pi_r[r] * np.prod( kappa_c.dot(g_cj))

            sum_r += zHat_ir[i, r]

        zHat_ir[i, :] /= sum_r

    g_ri = np.zeros((R, i), np.float64)
    xHat_jc = np.zeros((p, C), np.float64)
    for j in range(p):

        sum_c = 0.

        for c in range(C):

            for r in range(R):
                for i in range(n):
                    g_ri[c, j] = theta_rc[r, c]**data[i, j] * (1. - theta_rc[r, c])**(1. - data[i, j])

            xHat_jc[j, c] = kappa_c[c] * np.prod( pi_r.dot(g_ri))

            sum_c += xHat_jc[j, c]

        xHat_jc[j, :] /= sum_c

    
    return {'z_ir': zHat_ir, 'x_jc': xHat_jc}



def cluster_bi_bernoulli(data, R=2, C=2, maxiter=100, max_diff=1.e-6, seed=123):

    np.random.seed(seed)

    n = data.shape[0]
    p = data.shape[1]

    # randomly assign 1 for each row
    z_ir = create_random_z_ig(n, R)

    # randomly assign 1 to each column
    x_jc = create_random_z_ig(p, C)

    diff = float('inf')
    iteration = 0
    while iteration < maxiter and diff > max_diff:

        res = m_step(data, z_ir, x_jc)

        res2 = e_step_bernoulli(data, res['pi_r'], res['kappa_c'], res['theta_rc'])

        diff = np.sum(abs(z_ir - res2['z_ir'])) + np.sum(abs(x_jc - res2['x_jc']))

        z_ir = res2['z_ir']
        x_jc = res2['x_jc']

        iteration += 1

    if iteration >= maxiter:
        print(f'Warning: reached max number of iterations {maxiter}! diff = {diff}')

    res['iterations'] = iteration
    res['diff'] = diff

    return res


