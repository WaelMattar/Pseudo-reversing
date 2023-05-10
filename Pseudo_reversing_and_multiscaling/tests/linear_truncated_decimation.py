import numpy as np
import Even_singular_pyramid.Reversing.displacement_functions as uf
import sys


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

    # decimation[np.abs(decimation) < truncation_threshold] = 0
    # decimation = np.trim_zeros(decimation)
    argmax = 100
    decimation = decimation[int(argmax - truncation_size / 2 + 1): int(argmax + truncation_size / 2 + 1)]
    return list(decimation / sum(decimation))


if __name__ == '__main__':
    xis = list(np.linspace(0.0001, 3, 11))
    truncation_size = 11
    alpha, support = uf.least_squares_mask_and_support(d=1, n=2)
    delta = [0] * int(truncation_size - 2)
    delta[int(len(delta)/2)] = 1
    # alpha, support = [1/8, 1/2, 6/8, 1/2, 1/8], [-2, -1, 0, 1, 2]
    for xi in xis:
        new_alpha = uf.pseudo_reversing(mask=alpha.copy(), support=support, xi=xi)
        even_alpha = [new_alpha[support.index(k)] for k in support if k % 2 == 0]
        check_gamma = gamma(mask=new_alpha, support=support, truncation_size=truncation_size)
        convolution = np.convolve(check_gamma, even_alpha, mode='valid')
        error = np.linalg.norm(np.subtract(delta, convolution), ord=1)
        perturbation = np.linalg.norm(np.subtract(new_alpha, alpha), ord=np.infty)
        g_norm = np.linalg.norm(check_gamma, ord=1)
        print('xi = ' + str(np.round(xi, 4)) + '\tperturbation = ' + str(np.round(perturbation, 4))
              + '\tdelta error = ' + str(np.round(error, 4)) + '\tgamma norm = ' + str(np.round(g_norm, 4)))
