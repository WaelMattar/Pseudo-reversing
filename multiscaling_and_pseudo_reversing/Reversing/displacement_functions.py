from functools import reduce
import numpy as np
import sympy as syms
import sys


def mask_intertwine(even: list, odd: list):
    return [even[int(index / 2)] if index % 2 == 0 else odd[int(np.floor(index / 2))] for index in range(0, len(odd) + len(even))]


def least_squares_mask_and_support(d: float, n: float):
    if d > 2 * n or d * n == 1:
        sys.exit('The polynomial degree is larger than the number of data points to fit!')
    A = np.vander([2 * j for j in range(-n + 1, n)], N=d + 1, increasing=True)
    A_hat = np.vander([2 * j + 1 for j in range(-n, n)], N=d + 1, increasing=True)
    return mask_intertwine(np.flip(np.linalg.pinv(A_hat)[0]), np.flip(np.linalg.pinv(A)[0])), [k for k in range(-2*n + 1, 2*n)]


def pseudo_reversing(mask: list, support: list, xi: float):
    even_mask = [mask[support.index(k)] for k in support if k % 2 == 0]
    odd_mask = [mask[support.index(k)] for k in support if k % 2 == 1]
    roots = list(np.roots(np.flip(even_mask)))

    # For pseudo reversing reversible schemes: use the following 2 lines instead
    # on_unit_disc = [root for root in roots if np.abs(root) - 1 > 1e-8]
    # roots = [root for root in roots if np.abs(root) - 1 < 1e-8]

    on_unit_disc = [root for root in roots if np.abs(np.abs(root) - 1) < 1e-8]
    roots = [root for root in roots if np.abs(np.abs(root) - 1) > 1e-8]
    if len(on_unit_disc) == 0:
        return mask

    new_roots = [(1 + xi) * root for root in on_unit_disc]
    roots = roots + new_roots

    z = syms.Symbol('z')
    new_symbol = syms.expand(reduce(np.multiply, [z - root for root in roots]))
    new_symbol = syms.Poly(new_symbol / new_symbol.subs('z', 1))
    new_even_mask = [np.real(complex(coefficient)) for coefficient in new_symbol.all_coeffs()]
    new_even_mask = list(np.flip(new_even_mask))

    zero_pad = int((np.abs(len(new_even_mask) - len(odd_mask)) - 1) / 2)
    odd_mask = [0] * zero_pad + odd_mask + [0] * zero_pad
    if support[0] % 2 == 0:
        new_mask = [new_even_mask[int(k/2)] if k % 2 == 0 else odd_mask[int(k/2)] for k in range(len(support))]
    else:
        new_mask = [new_even_mask[int(k/2)] if k % 2 == 1 else odd_mask[int(k/2)] for k in range(len(support))]
    return new_mask


def gamma(mask: list, support: list, truncation_size='full'):
    padded_vector_size = 200
    even_mask = [mask[support.index(k)] for k in support if k % 2 == 0]

    if np.abs(sum(even_mask) - 1) > 1e-8:
        sys.exit('The even mask does not add to 1!')

    zero_pad = ([0] * int((padded_vector_size / 2) - np.ceil(len(even_mask) / 2)))
    padded_vector = zero_pad + even_mask + zero_pad
    fft = np.fft.fft(padded_vector)
    decimation = np.real(np.fft.ifft(np.reciprocal(fft)))

    if truncation_size == 'full':
        return decimation

    argmax = 100
    decimation = decimation[int(argmax - truncation_size / 2 + 1): int(argmax + truncation_size / 2 + 1)]
    return list(decimation / sum(decimation))
