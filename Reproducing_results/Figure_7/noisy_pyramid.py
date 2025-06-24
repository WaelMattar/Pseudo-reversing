from matplotlib.ticker import FormatStrFormatter
import Pseudo_reversing_and_multiscaling.Reversing.displacement_functions as df
import Pseudo_reversing_and_multiscaling.Linear.linear_functions as uf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
np.random.seed(1337)

n = 2
d = 1
xi = 1.4
truncation_size = 11
level_of_sampling = 6
pyramid_layers = 4

# operators
alpha, support = df.least_squares_mask_and_support(d=d, n=n)
new_alpha = df.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
gamma = df.gamma(mask=new_alpha, support=support, truncation_size=truncation_size)
g_norm = np.linalg.norm(df.gamma(mask=new_alpha, support=support), ord=1)
check_if_delta = np.convolve(gamma, [new_alpha[k] for k in range(len(new_alpha)) if support[k] % 2 == 0], 'same')
perturbation = np.linalg.norm(np.subtract(np.asarray(alpha), np.asarray(new_alpha)), ord=np.infty)
print('The original scheme is: ', alpha)
print('The approximated scheme is: ', new_alpha)
print('The infinity norm of the perturbation for xi = ' + str(xi) + ' is ' + str(perturbation))
print('The l1 norm of gamma (full) is: ', g_norm)

# pyramid
samples = uf.test_function(uf.dyadic_grid(0, 10, resolution=level_of_sampling + 1))
samples = samples + np.random.normal(0, 0.01, samples.shape)
pyramid = uf.pyramid(sequence=samples,
                     alpha=alpha,
                     alpha_support=support,
                     gamma=gamma,
                     layers=pyramid_layers)

# compressing
compressed = uf.pyramid_compress(pyramid=pyramid)

# print synthesis error
synthesis = uf.inverse_pyramid(pyramid=compressed,
                               alpha=alpha,
                               alpha_support=support)

diff = np.subtract(np.asarray(synthesis), np.asarray(samples))
print('The synthesis error is ' + str(np.linalg.norm(diff, ord=np.infty)))

fig, ax = plt.subplots(pyramid_layers, sharex='col', sharey='row', figsize=(8, 7))
for scale in range(1, pyramid_layers + 1):
    ax[scale - 1].scatter(uf.dyadic_grid(0, 10, resolution=level_of_sampling - pyramid_layers + scale + 1),
                          np.abs(pyramid[scale]),
                          color='k', s=5)
    ax[scale - 1].spines['top'].set_visible(False)
    ax[scale - 1].spines['right'].set_visible(False)
    ax[scale - 1].set_ylabel('scale ' + str(scale + level_of_sampling - pyramid_layers), fontsize=14)
    ax[scale - 1].set_ylim(0, np.max(np.abs(pyramid[scale])) + 0.005, emit=True)
    ax[scale - 1].set_xlim(min(uf.dyadic_grid(0, 10, resolution=scale + 1)),
                           max(uf.dyadic_grid(0, 10, resolution=scale + 1)), emit=True)
    ax[scale - 1].set_yticks([np.max(np.abs(pyramid[scale]))])
    ax[scale - 1].tick_params(axis='both', which='major', labelsize=13)
    ax[scale - 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[scale - 1].tick_params(axis='both', which='major', labelsize=16)

plt.figure(2, figsize=(8, 7))
plt.plot(uf.dyadic_grid(0, 10, resolution=level_of_sampling + 1), samples, c='k', linewidth=2, label='Analyzed')
plt.plot(uf.dyadic_grid(0, 10, resolution=level_of_sampling + 1), synthesis, c='b', linestyle='dashdot', linewidth=2, label='Synthesized')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(prop={'size': 16}, loc='upper right')

plt.figure(3)
plt.plot(uf.dyadic_grid(0, 10, resolution=level_of_sampling + 1), diff, c='g', linewidth=2)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()
