from matplotlib.ticker import FormatStrFormatter
import Even_singular_pyramid.Reversing.displacement_functions as df
import linear_functions as uf
import numpy as np
import matplotlib.pyplot as plt

n = 2
d = 1
xi = 0.5661
truncation_size = 11
number_of_layers = 5
noise_variance = 0.000000000003
threshold = 1.1

# operators
alpha, support = df.least_squares_mask_and_support(d=d, n=n)
new_alpha = df.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
gamma = df.gamma(mask=new_alpha, support=support, truncation_size=truncation_size)

# pyramid
samples = uf.test_function(uf.dyadic_grid(0, 10, resolution=number_of_layers + 1))
noisy_samples = uf.add_noise(samples=samples, noise_variance=noise_variance)
pyramid = uf.pyramid(sequence=noisy_samples,
                     alpha=alpha,
                     alpha_support=support,
                     gamma=gamma,
                     layers=number_of_layers)

# thresholding
sparse_pyramid = uf.pyramid_threshold(pyramid=pyramid.copy(), threshold=threshold)

# print synthesis error
synthesis = uf.inverse_pyramid(pyramid=sparse_pyramid.copy(),
                               alpha=alpha,
                               alpha_support=support)

# denoising effect
print('Noise before reduction = ', np.linalg.norm(np.subtract(samples, noisy_samples), ord=1))
print('Noise after reduction = ', np.linalg.norm(np.subtract(samples, synthesis), ord=1))

# plots
fig, ax = plt.subplots(number_of_layers, sharex='col', sharey='row', figsize=(8, 6))
for scale in range(1, number_of_layers + 1):
    ax[scale - 1].scatter(uf.dyadic_grid(0, 10, resolution=scale + 1), np.abs(pyramid[scale]), color='k', s=15)
    ax[scale - 1].spines['top'].set_visible(False)
    ax[scale - 1].spines['right'].set_visible(False)
    ax[scale - 1].set_ylabel('scale  ' + str(scale), fontsize=14)
    ax[scale - 1].set_ylim(0, np.max(np.abs(pyramid[scale])) + 0.07, emit=True)
    ax[scale - 1].set_xlim(min(uf.dyadic_grid(0, 10, resolution=scale + 1)),
                           max(uf.dyadic_grid(0, 10, resolution=scale + 1)), emit=True)
    ax[scale - 1].set_yticks([np.max(np.abs(pyramid[scale]))])
    ax[scale - 1].tick_params(axis='both', which='major', labelsize=13)
    ax[scale - 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.figure(2, figsize=[8, 6])
plt.scatter(uf.dyadic_grid(0, 10, resolution=number_of_layers + 1), noisy_samples, c='r', alpha=0.7, s=5,  label='noisy samples')
plt.plot(uf.dyadic_grid(0, 10, resolution=number_of_layers + 1), samples, c='k', alpha=0.8, markersize=15, label='ground truth')
plt.plot(uf.dyadic_grid(0, 10, resolution=number_of_layers + 1), synthesis, alpha=1, markersize=10, c='b', label='denoised')
plt.legend()
plt.show()
