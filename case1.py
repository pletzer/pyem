import pyem

data = pyem.read_table('data/case1_data.txt')
groups = {e for e in pyem.read_table('data/case1_groups.txt').flat}
G = len(groups)
theta_exact = pyem.read_table('data/case1_theta_matrix.txt')

res = pyem.cluster_bernoulli(data, G=G, maxiter=100, max_diff=1.e-10, seed=123)

print(f"""
Number of groups: {G}
theta: {res['theta_gj']}
p_ig: {res['p_ig']}
pi_g: {res['pi_g']}
	""")
