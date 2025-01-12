import displacement_functions as uf
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='serif')


def loss_function(alpha: list, support: list, xi: float):
    even_alpha = [alpha[support.index(k)] for k in support if k % 2 == 0]
    new_mask = uf.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
    gamma = uf.gamma(mask=new_mask, support=support)
    even_mask = [new_mask[k] for k in range(len(new_mask)) if support[k] % 2 == 0]
    delta = [0] * int(200 - len(even_alpha))
    delta[int(100 - np.floor(len(even_alpha)/2))] = 1
    return np.linalg.norm(np.subtract(delta, np.convolve(gamma, even_alpha, 'valid')), ord=2), np.linalg.norm(np.subtract(even_mask, even_alpha), ord=1)


n = 2
d = 1
lower_xi = 1e-10
upper_xi = 4
xis = list(np.linspace(lower_xi, upper_xi, 200))
alpha, support = uf.least_squares_mask_and_support(d=d, n=n)
loss = []

# calculating losses
for xi in xis:
    loss.append(loss_function(alpha, support, xi))
    print('xi = ' + str(xi))
    print('\t loss = ', loss[-1])

# loss plot
plt.figure(1, figsize=(8, 7))
plt.plot(xis, [loss[k][0] for k in range(len(loss))], alpha=1, label=r'Convolutional error', linewidth=3)
plt.plot(xis, [loss[k][1] for k in range(len(loss))], alpha=1, linestyle='dashed', label=r'Mask perturbation', linewidth=3)
plt.xlim([lower_xi, upper_xi])
plt.ylim(bottom=0)
plt.xticks(lower_xi + np.linspace(lower_xi, upper_xi, 11))
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel(r'$\xi$', fontsize=15)
plt.legend(prop={'size': 16})
plt.savefig('Figures/loss_analysis.pdf', format='pdf')
plt.show()
