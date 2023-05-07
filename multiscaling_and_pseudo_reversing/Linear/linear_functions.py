import numpy as np
import sys
import operator
from functools import reduce
from scipy import signal
import copy


def dyadic_grid(left_boundary: float, right_boundary: float, resolution: int, dilation_factor: int = 2):
    return np.linspace(left_boundary, right_boundary,
                       (right_boundary - left_boundary) * dilation_factor ** (resolution - 1) + 1)


def test_function(x):
    # return np.exp(-1 * (x-5)**2)
    return np.real(signal.morlet(M=len(x), w=5, s=1))
    # return signal.bspline(x=x-5, n=1)
    # return np.sinc(4/(np.pi) * (x-5))
    # return 1.4 * np.exp(-6 * (x-3) ** 2) - 1.4 * np.exp(-3 * (x-5) ** 2) + 1 * np.exp(-2 * (x-7) ** 2)


def add_noise(samples: list, std: float):
    r = np.random.RandomState(42)
    noise = r.normal(loc=0.0, scale=std, size=len(samples))
    return np.add(samples, noise), noise


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


def subdivision_scheme_multiple_times(mask: list, support: list, sequence: list, n: int):
    if n == 1:
        return subdivision_scheme(mask, support, sequence)
    else:
        return subdivision_scheme_multiple_times(mask, support, subdivision_scheme(mask, support, sequence), n - 1)


def downsample(sequence: list):
    return [sequence[int(2 * k)] for k in range(int(len(sequence) / 2 + 1))]


def downsample_multiple_times(sequence: list, n: int):
    if n == 1:
        return downsample(sequence)
    else:
        return downsample_multiple_times(downsample(sequence), n - 1)


def decimation(mask: list, sequence: list):
    sequence = downsample(sequence)
    means = []
    for k in range(len(sequence) - len(mask) + 1):
        points = sequence[k: k + len(mask)]
        means.append(np.dot(a=mask, b=points))

    edges = int(len(sequence) - len(means))
    means = np.pad(means, (int(np.ceil(edges/2)), int(np.floor(edges/2))), mode='edge')
    return means


def decomposition(sequence: list, alpha: list, alpha_support: list, gamma: list):
    decimated = decimation(gamma, sequence)
    refined = subdivision_scheme(mask=alpha, support=alpha_support, sequence=decimated)
    detail_coefficients = [sequence[k] - refined[k] for k in range(len(refined))]
    return [decimated, detail_coefficients]


def reconstruct(sequence: list, detail_coefficients: list, alpha: list, alpha_support: list):
    refined = subdivision_scheme(mask=alpha, support=alpha_support, sequence=sequence)
    return [detail_coefficients[k] + refined[k] for k in range(len(refined))]


def pyramid(sequence: list, alpha: list, alpha_support: list, gamma: list, layers: int):
    if layers < 1:
        sys.exit('Level is less than one!')
    representation = decomposition(sequence, alpha, alpha_support, gamma)
    for _ in range(layers - 1):
        decompose = decomposition(representation[0], alpha, alpha_support, gamma)
        representation = decompose + representation[1:]
    return representation


def inverse_pyramid(pyramid: list, alpha: list, alpha_support: list):
    level = len(pyramid)
    if level == 1:
        sys.exit('Pyramid only contains the coarse approximation!')
    reconstructed = reconstruct(pyramid[0], pyramid[1], alpha, alpha_support)
    for scale in range(2, level):
        reconstructed = reconstruct(reconstructed, pyramid[scale], alpha, alpha_support)
    return reconstructed


def pyramid_threshold(pyramid: list, threshold: float):
    for scale in range(1, len(pyramid)):
        for k in range(len(pyramid[scale])):
            if np.abs(pyramid[scale][k]) < threshold:
                pyramid[scale][k] = 0
    return pyramid


def pyramid_compress(pyramid: list):
    compressed_pyramid = copy.deepcopy(pyramid)
    for level in range(len(compressed_pyramid)-1):
        for k in range(int(np.floor(len(compressed_pyramid[-level-1]) / 2))):
             compressed_pyramid[-level-1][2*k + 1] = 0
    return compressed_pyramid
