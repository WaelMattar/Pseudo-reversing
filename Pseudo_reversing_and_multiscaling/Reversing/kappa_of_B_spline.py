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


B_spline_order = 6
xis = list(np.linspace(0, 1.3, 14))

support = list(np.linspace(-np.ceil(B_spline_order / 2), np.floor(B_spline_order / 2) + 1, B_spline_order + 2))
B_spline = [comb(B_spline_order + 1, k) for k in range(0, B_spline_order + 2)]
B_spline = [2 * coeff / np.sum(B_spline) for coeff in B_spline]

# Make sure to change the condition for root manipulations in uf.pseudo_reversing!
for xi in xis:
    new_mask = uf.pseudo_reversing(mask=B_spline.copy(), support=support.copy(), xi=xi)
    reversibility_condition = kappa(new_mask, support)
    print('The reversibility condition for B-spline of degree ' + str(B_spline_order) + ' for xi=' + str(xi) + ' is '
          + str(reversibility_condition))
