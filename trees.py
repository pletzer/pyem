import pyem
import numpy as np

data = np.array(np.loadtxt('data/binary.txt'), np.float64)

G = 2
res = pyem.cluster_bernoulli(data, G=G, maxiter=1000, max_diff=1.e-10, seed=234)

print(f"""
Number of groups: {G}
theta: {res['theta_gj']}
p_ig: {res['p_ig']}
pi_g: {res['pi_g']}
	""")
