import Pseudo_reversing_and_multiscaling.Reversing.displacement_functions as df
import so3_functions as uf
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
samples = uf.SO3_generate_smooth_sequence(resolution=number_of_layers)
pyramid = uf.SO3_pyramid(sequence=samples,
                         alpha=alpha,
                         alpha_support=support,
                         gamma=gamma,
                         layers=number_of_layers)

# detail coefficients scaling
factor = 1.4
norms = uf.SO3_pyramid_norms(pyramid=pyramid.copy())
for level in range(1, number_of_layers + 1):
    threshold = np.quantile(norms[level], q=0.80)
    pyramid[level] = [factor * pyramid[level][k]
                      if norms[level][k] > threshold else pyramid[level][k] for k in range(len(pyramid[level]))]

# synthesis
synthesis = uf.SO3_inverse_pyramid(pyramid=pyramid,
                                   alpha=alpha,
                                   alpha_support=support)
riemannian_distances = [uf.SO3_Riemannian_dist(A=samples[k], B=synthesis[k]) for k in range(len(samples))]
print('The synthesis error is ', max(riemannian_distances))

# plot curves
basis_matrix = np.diag([1] * 3)
for _ in range(number_of_layers - 3):
    samples = uf.downsample(samples)
loc = np.linspace(0, 10, len(samples))
scale = 10
length = 9
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for rotation, x in zip(samples, loc):
    result = np.matmul(rotation, basis_matrix)
    ax.quiver(scale * x, 0, 0, result[0, 0], result[1, 0], result[2, 0], length=length, color='blue')
    ax.quiver(scale * x, 0, 0, result[0, 1], result[1, 1], result[2, 1], length=length, color='red')
    ax.quiver(scale * x, 0, 0, result[0, 2], result[1, 2], result[2, 2], length=length, color='green')
plt.axis('off')
uf.set_axes_equal(ax)
ax.view_init(azim=60, elev=90)
plt.savefig('Figures/before_enhancement.eps', format='eps', bbox_inches='tight', pad_inches=0)

# contrast
fig = plt.figure(2, figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for _ in range(number_of_layers - 3):
    synthesis = uf.downsample(synthesis)
    riemannian_distances = uf.downsample(riemannian_distances)
max_error = max(riemannian_distances)
for rotation, x, error in zip(synthesis, loc, riemannian_distances):
    result = np.matmul(rotation, basis_matrix)
    ax.quiver(scale * x, 0, 0, result[0, 0], result[1, 0], result[2, 0], length=length, color='blue', alpha=max(error/max_error, 0.03))
    ax.quiver(scale * x, 0, 0, result[0, 1], result[1, 1], result[2, 1], length=length, color='red', alpha=max(error/max_error, 0.03))
    ax.quiver(scale * x, 0, 0, result[0, 2], result[1, 2], result[2, 2], length=length, color='green', alpha=max(error/max_error, 0.03))
plt.axis('off')
uf.set_axes_equal(ax)
ax.view_init(azim=60, elev=90)
plt.savefig('Figures/after_enhancement.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.show()
