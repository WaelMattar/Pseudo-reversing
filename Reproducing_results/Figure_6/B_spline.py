import matplotlib.pyplot as plt
import Pseudo_reversing_and_multiscaling.Reversing.displacement_functions as uf
from math import comb
import numpy as np

min_order = 4
max_order = 7

plt.figure(1, figsize=(8, 6))
theta = np.linspace(-np.pi, np.pi, 100)
plt.plot(1 * np.cos(theta), 1 * np.sin(theta), color='k', linewidth=3)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-3, 2)
plt.axis('off')
for order in range(min_order, max_order):
    support = list(np.linspace(-np.ceil(order/2), np.floor(order/2) + 1, order + 2))
    B_spline = [comb(order + 1, k) for k in range(0, order + 2)]
    even_mask = [B_spline[support.index(k)] for k in support if k % 2 == 0]
    roots = np.roots(p=np.flip(even_mask))
    plt.scatter([ele.real for ele in roots], [ele.imag for ele in roots], label='B-spline order = ' + str(order), s=40+160*(order-min_order)/max_order, marker='o')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(prop={'size': 16}, loc='upper left')
plt.xlim([-3.4, 1.2])
plt.ylim([-1.2, 1.2])

plt.figure(2, figsize=(8, 6))
for order in range(min_order, max_order):
    support = list(np.linspace(-np.ceil(order/2), np.floor(order/2) + 1, order + 2))
    B_spline = [comb(order + 1, k) for k in range(0, order + 2)]
    B_spline = [2 * coeff / np.sum(B_spline) for coeff in B_spline]
    gamma = uf.gamma(mask=B_spline, support=support)
    plt.plot(np.linspace(-len(gamma)/2+1, len(gamma)/2, len(gamma)), gamma, label='B-spline order = ' + str(order), linewidth=max_order + 1 - order)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(prop={'size': 16}, loc='upper right')
plt.xlim([-5, 15])
plt.xticks(range(-5, 16, 5))
plt.show()
