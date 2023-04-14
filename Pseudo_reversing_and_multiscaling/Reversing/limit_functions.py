import displacement_functions as duf
import Even_singular_pyramid.Linear.linear_functions as uf
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='serif')

n = 2
d = 1
xis = list(np.linspace(0.0001, .4, 4))
alpha, support = duf.least_squares_mask_and_support(d=d, n=n)
even_alpha = [alpha[support.index(k)] for k in support if k % 2 == 0]

delta = np.array([0] * len(uf.dyadic_grid(-10, 10, 1)))
delta[int(len(delta) / 2)] = 1

plt.figure(1, figsize=(8, 7))
plt.scatter(uf.dyadic_grid(-10, 10, 1), delta, label='Delta', color='k')
for xi in xis:
    new_mask = duf.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
    refined = uf.subdivision_scheme_multiple_times(mask=new_mask, support=support, sequence=delta, n=5)
    plt.plot(uf.dyadic_grid(-10, 10, resolution=5 + 1), refined, alpha=8/4*xi + 0.2, label=r'$\xi = $' + str(xi), linewidth=3)
plt.ylim(-0.05, 0.4)
plt.xlim(-4, 4)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(prop={'size': 16}, loc='upper right')
plt.savefig('Figures/limit_functions.pdf', format='pdf')
plt.legend()
plt.show()
