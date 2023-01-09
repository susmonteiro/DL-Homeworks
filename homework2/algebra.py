import numpy as np
import colorama
from colorama import Fore, Style

colorama.init()

w_M = 3
w_N = 2

x_H = 4
x_W = 3


h_prime = x_H - w_M + 1
w_prime = x_W - w_N + 1

M_height = h_prime * w_prime
M_width = x_H * x_W

w = np.random.randint(1, 10, size=(w_M, w_N))

M = np.zeros((M_height, M_width))

print(w)
print(M)

def compute_alpha(i):
    alpha = (x_W * (i // w_prime)) + (i % w_prime)
    # print("i = " + str(i) + ", alpha = " + str(alpha))
    return alpha

for i in range(M.shape[0]):
    alpha = compute_alpha(i)
    for j in range(M.shape[1]):
        if j >= alpha and ((j - alpha) % x_W) < w_N and ((j - alpha) // x_W) < w_M:
            M[i, j] = w[(j - alpha) // x_W, (j - alpha) % x_W]
            print(Fore.RED + str(int(M[i, j])) + Style.RESET_ALL, end=' ')
        else:
            print(int(M[i, j]), end=' ')
    print()

# TESTS
# W: 2x1    X: 3x2
# W: 2x2    X: 3x2      
# W: 2x2    X: 2x3
# W: 2x2    X: 4x3
# W: 2x3    X: 4x3
# W: 3x2    X: 4x3      INCORRECT
