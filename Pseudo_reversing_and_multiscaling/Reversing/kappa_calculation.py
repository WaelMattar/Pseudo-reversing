import displacement_functions as uf
from math import comb
import numpy as np
import sympy as syms


def kappa(mask: list, support: list):
    even_mask = [mask[support.index(k)] for k in support if k % 2 == 0]
    z = syms.Symbol('z')
    summand = [even_mask[index] * z ** index for index in range(0, len(even_mask))]
    even_symbol = syms.simplify(sum(summand))

    values = []
    unit_circle = [np.exp(complex(0, 1) * theta) for theta in np.linspace(0, 2*np.pi, 400)]
    for point in unit_circle:
        values.append(syms.simplify(even_symbol.subs('z', point)))

    abs_values = np.abs(values)
    supremum = np.max(abs_values)
    infimum = np.min(abs_values)
    if infimum < 10e-9:
        return np.inf
    kappa = supremum / infimum
    return kappa


n = 2
d = 1
xis = list(np.linspace(0, 1.5, 10))
alpha, support = uf.least_squares_mask_and_support(d=d, n=n)

print('The original even-singular mask is: ', alpha)
print('supported on: ', support)

for xi in xis:
    # root distinction
    new_mask = uf.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
    print('For xi = ' + str(xi) + ' The even regular mask is: ', new_mask)

    reversibility_condition = kappa(new_mask, support)
    print('The reversibility condition is: ', reversibility_condition)

for order in range(2, 8):
    support = list(np.linspace(-np.ceil(order / 2), np.floor(order / 2) + 1, order + 2))
    B_spline = [comb(order + 1, k) for k in range(0, order + 2)]
    B_spline = [2 * coeff / np.sum(B_spline) for coeff in B_spline]
    reversibility_condition = kappa(B_spline, support)
    print('The reversibility condition for B-spline of degree ' + str(order) + ' is ' + str(reversibility_condition))
