from matplotlib.ticker import FormatStrFormatter
import Even_singular_pyramid.Reversing.displacement_functions as df
import linear_functions as uf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')

n = 2
d = 1
xi = 1.4
truncation_size = 11
level_of_sampling = 6
pyramid_layers = 6

# operators
alpha, support = df.least_squares_mask_and_support(d=d, n=n)
# alpha, support = [1/8, 1/2, 6/8, 1/2, 1/8], [-2, -1, 0, 1, 2]
new_alpha = df.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
gamma = df.gamma(mask=new_alpha, support=support, truncation_size=truncation_size)
g_norm = np.linalg.norm(df.gamma(mask=new_alpha, support=support), ord=1)
perturbation = np.linalg.norm(np.subtract(np.asarray(alpha), np.asarray(new_alpha)), ord=np.infty)
print('The original scheme is: ', alpha)
print('The approximated scheme is: ', new_alpha)
print('The infinity norm of the perturbation for xi = ' + str(xi) + ' is ' + str(perturbation))
print('The l1 norm of gamma (full) is: ', g_norm)

# pyramids
samples = uf.test_function(uf.dyadic_grid(0, 10, resolution=level_of_sampling + 1))
samples = uf.add_noise(samples=samples, noise_variance=0.8e+1)
pyramid = uf.pyramid(sequence=samples,
                     alpha=alpha,
                     alpha_support=support,
                     gamma=gamma,
                     layers=pyramid_layers)

pyramid_tilde = uf.pyramid(sequence=samples,
                           alpha=new_alpha,
                           alpha_support=support,
                           gamma=gamma,
                           layers=pyramid_layers)

# print synthesis error
synthesis = uf.inverse_pyramid(pyramid=pyramid,
                               alpha=alpha,
                               alpha_support=support)

synthesis_tilde = uf.inverse_pyramid(pyramid=pyramid_tilde,
                                     alpha=new_alpha,
                                     alpha_support=support)

# print errors
for k in range(1, len(pyramid)):
    error = np.linalg.norm(np.subtract(np.asarray(pyramid[k]), np.asarray(pyramid_tilde[k])), ord=np.infty)
    print('The sup error in layer number ' + str(k-1) + ' is ' + str(error))
print('The synthesis difference is ' + str(np.linalg.norm(np.subtract(np.asarray(synthesis), np.asarray(synthesis_tilde)), ord=np.infty)))

# plot
fig, ax = plt.subplots(pyramid_layers, sharex='col', sharey='row', figsize=(8, 6))
for scale in range(1, pyramid_layers + 1):
    ax[scale - 1].scatter(uf.dyadic_grid(0, 10, resolution=level_of_sampling - pyramid_layers + scale + 1),
                          np.abs(pyramid[scale]),
                          color='k', s=5)
    ax[scale - 1].scatter(uf.dyadic_grid(0, 10, resolution=level_of_sampling - pyramid_layers + scale + 1),
                          np.abs(pyramid_tilde[scale]),
                          color='r', s=5)
    ax[scale - 1].spines['top'].set_visible(False)
    ax[scale - 1].spines['right'].set_visible(False)
    ax[scale - 1].set_ylabel('scale ' + str(scale), fontsize=14)
    ax[scale - 1].set_ylim(0, np.max(np.abs(pyramid[scale] + pyramid_tilde[scale])) + 0.005, emit=True)
    ax[scale - 1].set_xlim(min(uf.dyadic_grid(0, 10, resolution=scale + 1)),
                           max(uf.dyadic_grid(0, 10, resolution=scale + 1)), emit=True)
    ax[scale - 1].set_yticks([np.max(np.abs(pyramid[scale]))])
    ax[scale - 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[scale - 1].tick_params(axis='both', which='major', labelsize=16)
plt.show()
