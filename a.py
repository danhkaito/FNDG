import numpy as np
rand_arr = np.asarray([j for j in range(100)])
np.random.shuffle(rand_arr)
print(rand_arr)