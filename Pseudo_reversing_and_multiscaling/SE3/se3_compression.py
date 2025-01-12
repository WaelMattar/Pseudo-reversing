import Pseudo_reversing_and_multiscaling.Reversing.displacement_functions as df
import se3_functions as uf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
samples = uf.SE3_generate_smooth_sequence(resolution=level_of_sampling)
pyramid = uf.SE3_pyramid(sequence=samples,
                         alpha=alpha,
                         alpha_support=support,
                         gamma=gamma,
                         layers=pyramid_layers)

# zero even details + compression by ratio
compressed_1 = uf.pyramid_zero_even_details(pyramid=pyramid)
compressed = uf.pyramid_compress_ratio(pyramid=compressed_1, ratio=0.99)

# synthesis
synthesis = uf.SE3_inverse_pyramid(pyramid=compressed,
                                   alpha=alpha,
                                   alpha_support=support)
relative_riemannian_distances = [uf.SE3_Riemannian_dist(A=samples[k], B=synthesis[k]) for k in range(len(samples))]
print('The synthesis error is ', np.median(relative_riemannian_distances))

# plot error histogram
plt.figure(99, figsize=(8, 6))
sns.histplot(relative_riemannian_distances, bins=50, edgecolor='black', color='blue', kde=True)
plt.xlabel('Riemannian distance', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Figures/Riemannian_distance_histogram.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# plot curves
for _ in range(level_of_sampling - 3):
    samples = uf.downsample(samples)
rotation = [samples[k][0:3, 0:3] for k in range(len(samples))]
location = [samples[k][0:3, -1] for k in range(len(samples))]
basis_matrix = np.diag([1] * 3)
length = .7
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for rot, loc in zip(rotation, location):
    result = np.matmul(rot, basis_matrix)
    ax.quiver(loc[0], loc[1], loc[2], result[0, 0], result[1, 0], result[2, 0], length=length, color='blue')
    ax.quiver(loc[0], loc[1], loc[2], result[0, 1], result[1, 1], result[2, 1], length=length, color='red')
    ax.quiver(loc[0], loc[1], loc[2], result[0, 2], result[1, 2], result[2, 2], length=length, color='green')
# position
ax.plot3D([location[k][0] for k in range(len(location))], [location[k][1] for k in range(len(location))], [location[k][2] for k in range(len(location))], 'k')
plt.axis('on')
uf.set_axes_equal(ax)
ax.view_init(azim=-150, elev=0)
plt.savefig('Figures/SE3_curve.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# compressed
for _ in range(level_of_sampling - 3):
    synthesis = uf.downsample(synthesis)
rotation = [synthesis[k][0:3, 0:3] for k in range(len(synthesis))]
location = [synthesis[k][0:3, -1] for k in range(len(synthesis))]
fig = plt.figure(2, figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for rot, loc in zip(rotation, location):
    result = np.matmul(rot, basis_matrix)
    ax.quiver(loc[0], loc[1], loc[2], result[0, 0], result[1, 0], result[2, 0], length=length, color='blue')
    ax.quiver(loc[0], loc[1], loc[2], result[0, 1], result[1, 1], result[2, 1], length=length, color='red')
    ax.quiver(loc[0], loc[1], loc[2], result[0, 2], result[1, 2], result[2, 2], length=length, color='green')
# position
ax.plot3D([location[k][0] for k in range(len(location))], [location[k][1] for k in range(len(location))], [location[k][2] for k in range(len(location))], 'k')
plt.axis('on')
uf.set_axes_equal(ax)
ax.view_init(azim=-150, elev=0)
plt.savefig('Figures/SE3_compression.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.show()
