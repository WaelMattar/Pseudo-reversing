import displacement_functions as uf
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='serif')

n = 2
d = 1
truncation_size = 11
xis = list(np.linspace(0.0001, .4, 4))
alpha, support = uf.least_squares_mask_and_support(d=d, n=n)
even_alpha = [alpha[support.index(k)] for k in support if k % 2 == 0]

plt.figure(1, figsize=(7, 6))
for xi in xis:
    new_mask = uf.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
    even_mask = [new_mask[support.index(k)] for k in support if k % 2 == 0]
    plt.scatter([-2, 0, 2], even_mask, label=r'$\xi = $' + str(np.round(xi, 4)), marker='_', s=2000 - 4000 * xi)
plt.xlim([-2.5, 2.5])
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(prop={'size': 16}, loc='upper right')
plt.savefig('Figures/mask_perturbation.eps', format='eps')
plt.show()
