import numpy as np
import matplotlib.pyplot as plt
import operator
import Even_singular_pyramid.Reversing.displacement_functions as uf
from functools import reduce


def function(x):
    # return np.sin(8*(x-5)) / (8*(x-5))
    return np.exp(-100 * (x-5) ** 2)
    # return np.random.normal(loc=0, scale=1, size=21).tolist()
    # return 1.4 * np.exp(-6 * (x-3) ** 2) - 0.7 * np.exp(-6 * (x-5) ** 2) + 1.4 * np.exp(-6 * (x-7) ** 2)


def dyadic_grid(left_boundary: float, right_boundary: float, resolution: int, dilation_factor: int = 2):
    return np.linspace(left_boundary, right_boundary,
                       (right_boundary - left_boundary) * dilation_factor ** (resolution - 1) + 1)


def mergeLists(lst1, lst2):
    return list(reduce(operator.add, zip(lst1, lst2)))


def subdivision_scheme(mask: list, support: list, sequence: list):
    even_mask = [mask[support.index(k)] for k in support if k % 2 == 0]
    odd_mask = [mask[support.index(k)] for k in support if k % 2 == 1]

    even_means = []
    for k in range(len(sequence) - len(even_mask) + 1):
        points = sequence[k: k + len(even_mask)]
        even_means.append(np.dot(a=even_mask, b=points))

    odd_means = []
    for k in range(len(sequence) - len(odd_mask) + 1):
        points = sequence[k: k + len(odd_mask)]
        odd_means.append(np.dot(a=odd_mask, b=points))

    means = mergeLists(odd_means, even_means) if len(even_mask) > len(odd_mask) else mergeLists(even_means, odd_means)
    edges = int(2 * len(sequence) - 1 - len(means))
    means = np.pad(means, (int(np.floor(edges/2)), int(np.ceil(edges/2))), mode='edge')
    return means


if __name__ == '__main__':
    samples = function(dyadic_grid(0, 10, resolution=2))
    alpha, support = uf.least_squares_mask_and_support(d=1, n=2)
    # alpha, support = [1/8, 1/2, 6/8, 1/2, 1/8], [-2, -1, 0, 1, 2]
    # alpha, support = [-1/16, 0, 9/16, 1, 9/16, 0, -1/16], [-3, -2, -1, 0, 1, 2, 3]

    n = 1
    refined = subdivision_scheme(alpha, support, sequence=samples)
    for _ in range(n-1):
        refined = subdivision_scheme(alpha, support, sequence=refined)

    plt.figure()
    plt.scatter(dyadic_grid(0, 10, resolution=n+2), refined, marker='o', c='k', s=4)
    plt.scatter(dyadic_grid(0, 10, resolution=2), samples, marker='*', c='r', s=20)
    plt.show()
