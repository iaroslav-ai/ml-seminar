from time import time
from skopt import forest_minimize
from skopt.benchmarks import hart6

bounds = [(0., 1.),] * 6
n_calls = 200

s = time()
res = forest_minimize(hart6, bounds, n_calls=n_calls, random_state=4)
print(time()-s)

bounds = [(0., 1.),] * 6
n_calls = 200

s = time()
res = forest_minimize(hart6, bounds, n_calls=n_calls, n_jobs=4, random_state=4)
print(time()-s)
