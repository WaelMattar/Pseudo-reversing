import displacement_functions as uf
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='serif')

n = 2
d = 1
truncation_size = 11
xis = list(np.linspace(0.0001, 0.4, 4))
alpha, support = uf.least_squares_mask_and_support(d=d, n=n)
even_alpha = [alpha[support.index(k)] for k in support if k % 2 == 0]
delta = [0] * int(200 - len(even_alpha))
delta[int(100 - np.floor(len(even_alpha) / 2))] = 1

plt.figure(1, figsize=(7, 6))
for xi in xis:
    new_mask = uf.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
    gamma = uf.gamma(mask=new_mask, support=support)
    even_mask = [new_mask[k] for k in range(len(new_mask)) if support[k] % 2 == 0]
    convolution = np.convolve(gamma, even_mask, 'valid')
    plt.scatter(np.linspace(-99, 97, 197), np.log10(convolution), marker='o', s=120 - 250*xi, label=r'$\xi = $' + str(np.round(xi, 4)))
# plt.ylabel(r'$\log_{10}(\gamma * \alpha_{ev})$', fontsize=16)
plt.xlim([-30, 30])
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(prop={'size': 16}, loc='upper right')
plt.savefig('Figures/convolutional_error.eps', format='eps', bbox_inches='tight', pad_inches=0)
plt.show()
