import Pseudo_reversing_and_multiscaling.Reversing.displacement_functions as duf
import Pseudo_reversing_and_multiscaling.Linear.linear_functions as uf
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='serif')

#  TODO: the limit functions are shifted! To fix this, sway "ceil" with "floor" in line 48 in Linear.linear_functions.py

n = 2
d = 1
xis = list(np.linspace(0.1, 0.3, 3))
alpha, support = duf.least_squares_mask_and_support(d=d, n=n)
even_alpha = [alpha[support.index(k)] for k in support if k % 2 == 0]

delta = np.array([0] * len(uf.dyadic_grid(-10, 10, 1)))
delta[int(len(delta) / 2)] = 1

plt.figure(1, figsize=(8, 7))
plt.scatter(uf.dyadic_grid(-10, 10, 1), delta, color='k')
for xi in xis:
    new_mask = duf.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
    refined = uf.subdivision_scheme_multiple_times(mask=new_mask, support=support, sequence=delta, n=5)
    plt.plot(uf.dyadic_grid(-10, 10, resolution=5 + 1), refined, label=r'$\xi = $' + str(xi), linewidth=4 - 10*xi)
plt.ylim(-0.05, 0.4)
plt.xlim(-4, 4)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(prop={'size': 16}, loc='upper right')
plt.legend()
plt.show()
