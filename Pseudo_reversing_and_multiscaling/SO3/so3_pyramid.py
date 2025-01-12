from matplotlib.ticker import FormatStrFormatter
import Pseudo_reversing_and_multiscaling.Reversing.displacement_functions as df
import so3_functions as uf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')

n = 2
d = 1
xi = 0.64
truncation_size = 11
level_of_sampling = 6
pyramid_layers = 4

# operators
alpha, support = df.least_squares_mask_and_support(d=d, n=n)
new_alpha = df.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
gamma = df.gamma(mask=new_alpha, support=support, truncation_size=truncation_size)

# pyramid
samples = uf.SO3_generate_smooth_sequence(resolution=level_of_sampling)
pyramid = uf.SO3_pyramid(sequence=samples,
                         alpha=alpha,
                         alpha_support=support,
                         gamma=gamma,
                         layers=pyramid_layers)

# synthesis
synthesis = uf.SO3_inverse_pyramid(pyramid=pyramid,
                                   alpha=alpha,
                                   alpha_support=support)
riemannian_distances = [uf.SO3_Riemannian_dist(A=samples[k], B=synthesis[k]) for k in range(len(samples))]
print('The synthesis error is ', max(riemannian_distances))

# plot detail coefficients
fig, ax = plt.subplots(pyramid_layers, sharex='col', sharey='row', figsize=(8, 7))
norms = uf.SO3_pyramid_norms(pyramid=pyramid)
for scale in range(1, pyramid_layers + 1):
    ax[scale - 1].scatter(uf.dyadic_grid(0, 10, resolution=level_of_sampling - pyramid_layers + scale + 1),
                          norms[scale], color='k', s=5)
    ax[scale - 1].spines['top'].set_visible(False)
    ax[scale - 1].spines['right'].set_visible(False)
    ax[scale - 1].set_ylabel('scale ' + str(scale + level_of_sampling - pyramid_layers), fontsize=14)
    ax[scale - 1].set_ylim(0, np.max(norms[scale]) + 0.005, emit=True)
    ax[scale - 1].set_xlim(-0.1, 10.1, emit=True)
    ax[scale - 1].set_yticks([np.max(norms[scale])])
    ax[scale - 1].tick_params(axis='both', which='major', labelsize=13)
    ax[scale - 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.savefig('Figures/SO3_pyramid.pdf', format='pdf')

# plot curve
basis_matrix = np.diag([1] * 3)
for _ in range(level_of_sampling - 3):
    samples = uf.downsample(samples)
loc = np.linspace(0, 10, len(samples))
scale = 10
length = 9
fig = plt.figure(2, figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for rotation, x in zip(samples, loc):
    result = np.matmul(rotation, basis_matrix)
    ax.quiver(scale * x, 0, 0, result[0, 0], result[1, 0], result[2, 0], length=length, color='blue')
    ax.quiver(scale * x, 0, 0, result[0, 1], result[1, 1], result[2, 1], length=length, color='red')
    ax.quiver(scale * x, 0, 0, result[0, 2], result[1, 2], result[2, 2], length=length, color='green')
plt.axis('off')
uf.set_axes_equal(ax)
ax.view_init(azim=60, elev=90)
plt.savefig('Figures/SO3_curve.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.show()
