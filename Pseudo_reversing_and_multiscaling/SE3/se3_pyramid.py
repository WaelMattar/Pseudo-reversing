from matplotlib.ticker import FormatStrFormatter
import Pseudo_reversing_and_multiscaling.Reversing.displacement_functions as df
import se3_functions as uf
import numpy as np
import matplotlib.pyplot as plt

n = 2
d = 1
xi = 0.64
truncation_size = 11
number_of_layers = 6

# operators
alpha, support = df.least_squares_mask_and_support(d=d, n=n)
new_alpha = df.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
gamma = df.gamma(mask=new_alpha, support=support, truncation_size=truncation_size)

# pyramid
samples = uf.SE3_generate_smooth_sequence(resolution=number_of_layers)
pyramid = uf.SE3_pyramid(sequence=samples,
                         alpha=alpha,
                         alpha_support=support,
                         gamma=gamma,
                         layers=number_of_layers)

# synthesis
synthesis = uf.SE3_inverse_pyramid(pyramid=pyramid,
                                   alpha=alpha,
                                   alpha_support=support)
riemannian_distances = [uf.SE3_Riemannian_dist(A=samples[k], B=synthesis[k]) for k in range(len(samples))]
print('The synthesis error is ', max(riemannian_distances))

# plot detail coefficients
fig, ax = plt.subplots(number_of_layers, sharex='col', sharey='row', figsize=(8, 6))
norms = uf.SE3_pyramid_norms(pyramid=pyramid)
for scale in range(1, number_of_layers + 1):
    ax[scale - 1].scatter(uf.dyadic_grid(0, 10, resolution=scale + 1), norms[scale], color='k', s=15)
    ax[scale - 1].spines['top'].set_visible(False)
    ax[scale - 1].spines['right'].set_visible(False)
    ax[scale - 1].set_ylabel('scale  ' + str(scale), fontsize=14)
    ax[scale - 1].set_ylim(0, np.max(norms[scale]) + 0, emit=True)
    ax[scale - 1].set_xlim(-0.1, 10.1, emit=True)
    ax[scale - 1].set_yticks([np.max(norms[scale])])
    ax[scale - 1].tick_params(axis='both', which='major', labelsize=13)
    ax[scale - 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# plot curve
for _ in range(number_of_layers - 3):
    samples = uf.downsample(samples)
rotation = [samples[k][0:3, 0:3] for k in range(len(samples))]
location = [samples[k][0:3, -1] for k in range(len(samples))]
basis_matrix = np.diag([1] * 3)
length = .7
fig = plt.figure(2, figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for rot, loc in zip(rotation, location):
    result = np.matmul(rot, basis_matrix)
    ax.quiver(loc[0], loc[1], loc[2], result[0, 0], result[1, 0], result[2, 0], length=length, color='blue')
    ax.quiver(loc[0], loc[1], loc[2], result[0, 1], result[1, 1], result[2, 1], length=length, color='red')
    ax.quiver(loc[0], loc[1], loc[2], result[0, 2], result[1, 2], result[2, 2], length=length, color='green')
# position
ax.plot3D([location[k][0] for k in range(len(location))], [location[k][1] for k in range(len(location))], [location[k][2] for k in range(len(location))], 'k')
plt.axis('off')
uf.set_axes_equal(ax)
ax.view_init(azim=-150, elev=0)
plt.show()
