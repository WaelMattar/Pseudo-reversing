import Pseudo_reversing_and_multiscaling.Reversing.displacement_functions as uf
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='serif')

n = 2
d = 1
xis = list(np.linspace(0.1, 0.3, 3))
alpha, support = uf.least_squares_mask_and_support(d=d, n=n)

print('The original even-singular mask is: ', alpha)
print('supported on: ', support)

# decimation coefficients plot
plt.figure(1, figsize=(8, 6))
# plt.title('Decimation coefficients for LS scheme with d = ' + str(d) + ' and n = ' + str(n))
for xi in xis:
    # root distinction
    new_mask = uf.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
    print('The even regular mask is: ', new_mask)
    print('supported on: ', support)

    # calculate gamma
    gamma = uf.gamma(mask=new_mask, support=support)

    # plot
    side_length = int(len(gamma) / 2)
    plt.plot(np.linspace(-side_length - 1, side_length - 1, len(gamma)), gamma, label=r'$\xi = $' + str(np.round(xi, 4)), linewidth=10*xi)
plt.xlim([-5, 25])
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(prop={'size': 16}, loc='upper right')

# roots plot
plt.figure(2, figsize=[8, 6])
plt.gca().set_aspect('equal', 'box')
plt.axis('off')
plt.xlim(-1.9, 1.9)
plt.ylim(-1.9, 1.9)

# circle
theta = np.linspace(-np.pi, np.pi, 100)
plt.plot(1 * np.cos(theta), 1 * np.sin(theta), color='k', linewidth=3)
# plt.scatter([0], [0], marker='o', s=5, c='k', alpha=0.5)

for xi in xis:
    new_mask = uf.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
    even_mask = [new_mask[k] for k in range(len(new_mask)) if support[k] % 2 == 0]
    roots = np.roots(np.flip(even_mask))
    plt.scatter([ele.real for ele in roots], [ele.imag for ele in roots], label=r'$\xi = $' + str(np.round(xi, 4)), marker='o', s=120-300*(xi-0.1))
plt.legend(prop={'size': 16}, loc='center right')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()
