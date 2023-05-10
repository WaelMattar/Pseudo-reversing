from scipy.linalg import expm, logm
import numpy as np
import sys
import operator
from functools import reduce


def mergeLists(lst1, lst2):
    return list(reduce(operator.add, zip(lst1, lst2)))


def SO3_Riemannian_dist(A, B):
    return 1 / np.sqrt(2) * np.linalg.norm(logm(np.matmul(np.transpose(A), B)), ord='fro')


def SO3_mean(A, B, t: float = 1 / 2):
    M = expm(t * logm(np.matmul(np.transpose(A), B)))
    return np.matmul(A, M)


def SO3_GIM(matrices: list, weights: list):
    if len(matrices) != len(weights) or np.abs(sum(weights) - 1) > 1e-10:
        sys.exit('Number of matrices and weights are incompatible OR weights do not sum to 1.')
    matrices = [M for _, M in sorted(zip(weights, matrices), key=lambda tup: tup[0], reverse=True)]
    weights = sorted(weights, reverse=True)
    if len(matrices) == 2:
        return SO3_mean(matrices[0], matrices[-1], t=weights[-1])
    else:
        reduced_matrices = matrices[:-1]
        reduced_weights = np.multiply(weights[:-1], 1 / (1 - weights[-1]))
        return SO3_mean(SO3_GIM(matrices=reduced_matrices, weights=reduced_weights), matrices[-1], t=weights[-1])


def subdivision_scheme(mask: list, support: list, sequence: list):
    even_mask = [mask[support.index(k)] for k in support if k % 2 == 0]
    odd_mask = [mask[support.index(k)] for k in support if k % 2 == 1]

    even_means = []
    for k in range(len(sequence) - len(even_mask) + 1):
        points = sequence[k: k + len(even_mask)]
        even_means.append(SO3_GIM(matrices=points, weights=even_mask))

    odd_means = []
    for k in range(len(sequence) - len(odd_mask) + 1):
        points = sequence[k: k + len(odd_mask)]
        odd_means.append(SO3_GIM(matrices=points, weights=odd_mask))

    means = mergeLists(odd_means, even_means) if len(even_mask) > len(odd_mask) else mergeLists(even_means, odd_means)
    edges = int(2 * len(sequence) - 1 - len(means))
    means = list(np.pad(means, ((int(np.floor(edges/2)), int(np.ceil(edges/2))), (0, 0), (0, 0)), mode='edge'))
    return means


def subdivision_scheme_multiple_times(mask: list, support: list, sequence: list, n: int):
    if n == 1:
        return subdivision_scheme(mask, support, sequence)
    else:
        return subdivision_scheme_multiple_times(mask, support, subdivision_scheme(mask, support, sequence), n - 1)


def decimation(mask: list, sequence: list):
    sequence = downsample(sequence)
    means = []
    for k in range(len(sequence) - len(mask) + 1):
        points = sequence[k: k + len(mask)]
        means.append(SO3_GIM(matrices=points, weights=mask))

    edges = int(len(sequence) - len(means))
    means = list(np.pad(means, ((int(np.floor(edges/2)), int(np.ceil(edges/2))), (0, 0), (0, 0)), mode='edge'))
    return means


def downsample(sequence: list):
    return [sequence[2 * k] for k in range(int(len(sequence) / 2) + 1)]


def SO3_exp(A, B):
    """
    :param A: skew-symmetric 3x3
    :param B: B is the base rotation matrix
    :return: rotation matrix exp_B(A). Can be computed as expm of the skew-symmetric matrix
     corresponding to the angle-axis representation of B.
    """
    d = np.sqrt(1 / 2 * np.trace(np.matmul(np.transpose(A), A)))
    # if d >= np.pi:
    #     sys.exit('Matrix is not in the injectivity radius!')
    res = np.eye(3) + np.sin(d) / d * A + (1 - np.cos(d)) / (d ** 2) * np.linalg.matrix_power(A, 2) if np.abs(
        d) > 1e-10 else np.eye(3)
    return np.matmul(res, B)


def SO3_log(A, B):
    """
    :param A: rotation matrix within the injectivity radius of B
    :param B: base-point rotation matrix
    :return: skew-symmetric 3x3 matrix log_B(A)
    """
    return logm(np.matmul(A, np.linalg.inv(B)))


def SO3_generate_smooth_sequence(resolution: int):
    alpha = [1 / 8, 1 / 2, 3 / 4, 1 / 2, 1 / 8]
    support = [-2, -1, 0, 1, 2]
    SO3 = []
    r = np.random.RandomState(40)
    for _ in range(4):
        A, _ = np.linalg.qr(r.randn(3, 3))
        SO3.append(A)
    SO3 = [SO3[0]] * 3 + [SO3[1]] * 3 + [SO3[2]] * 3 + [SO3[3]] * 2
    return subdivision_scheme_multiple_times(mask=alpha, support=support, sequence=SO3, n=resolution)


def SO3_decomposition(sequence: list, alpha: list, alpha_support: list, gamma: list):
    decimated = decimation(gamma, sequence)
    refined = subdivision_scheme(mask=alpha, support=alpha_support, sequence=decimated)
    detail_coefficients = [SO3_log(sequence[k], refined[k]) for k in range(len(refined))]
    return [decimated, detail_coefficients]


def SO3_reconstruct(sequence: list, detail_coefficients: list, alpha: list, alpha_support: list):
    refined = subdivision_scheme(mask=alpha, support=alpha_support, sequence=sequence)
    return [SO3_exp(detail_coefficients[k], refined[k]) for k in range(len(refined))]


def SO3_pyramid(sequence: list, alpha: list, alpha_support: list, gamma: list, layers: int):
    if layers < 1:
        sys.exit('Level is less than one!')
    representation = SO3_decomposition(sequence, alpha, alpha_support, gamma)
    for _ in range(layers - 1):
        decompose = SO3_decomposition(representation[0], alpha, alpha_support, gamma)
        representation = decompose + representation[1:]
    return representation


def SO3_pyramid_norms(pyramid: list):
    for scale in range(1, len(pyramid)):
        detail_coefficients_norms = [np.linalg.norm(pyramid[scale][k], ord='fro') for k in range(len(pyramid[scale]))]
        pyramid[scale] = detail_coefficients_norms
    return pyramid


def SO3_inverse_pyramid(pyramid: list, alpha: list, alpha_support: list):
    level = len(pyramid)
    if level == 1:
        sys.exit('Pyramid only contains the coarse approximation!')
    reconstructed = SO3_reconstruct(pyramid[0], pyramid[1], alpha, alpha_support)
    for scale in range(2, level):
        reconstructed = SO3_reconstruct(reconstructed, pyramid[scale], alpha, alpha_support)
    return reconstructed


def dyadic_grid(left_boundary: float, right_boundary: float, resolution: int, dilation_factor: int = 2):
    return np.linspace(left_boundary, right_boundary, (right_boundary - left_boundary) * dilation_factor ** (resolution - 1) + 1)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
