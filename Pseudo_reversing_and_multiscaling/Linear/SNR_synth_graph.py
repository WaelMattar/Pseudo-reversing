from matplotlib.ticker import FormatStrFormatter
import Pseudo_reversing_and_multiscaling.Reversing.displacement_functions as df
import linear_functions as uf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')

n = 2
d = 1
xi = 1.4
truncation_size = 11
level_of_sampling = 6
pyramid_layers = 6
SNR = []
synth_error = []

# operators
alpha, support = df.least_squares_mask_and_support(d=d, n=n)
# alpha, support = [1/8, 1/2, 6/8, 1/2, 1/8], [-2, -1, 0, 1, 2]
new_alpha = df.pseudo_reversing(mask=alpha.copy(), support=support.copy(), xi=xi)
gamma = df.gamma(mask=new_alpha, support=support, truncation_size=truncation_size)
g_norm = np.linalg.norm(df.gamma(mask=new_alpha, support=support), ord=1)

samples = uf.test_function(uf.dyadic_grid(0, 10, resolution=level_of_sampling + 1))
noise_levels = np.linspace(0, 20, 1000)
for k in noise_levels:
    noisy, noise = uf.add_noise(samples=samples.copy(), std=10 ** k)
    pyramid = uf.pyramid(sequence=noisy.copy(),
                         alpha=alpha,
                         alpha_support=support,
                         gamma=gamma,
                         layers=pyramid_layers)

    # print synthesis error
    synthesis = uf.inverse_pyramid(pyramid=pyramid,
                                   alpha=alpha,
                                   alpha_support=support)

    noise_L2 = np.linalg.norm(noise, ord=2)
    noisy_L2 = np.linalg.norm(noisy, ord=2)
    snr = np.log10(noisy_L2 / noise_L2)
    SNR.append(snr)
    diff = np.subtract(np.asarray(synthesis), np.asarray(noisy))
    synth_error.append(np.linalg.norm(diff, ord=np.infty))
    print('For SNR=' + str(snr) + ' the synthesis error is ' + str(np.linalg.norm(diff, ord=np.infty)))

plt.figure(1, figsize=(8, 6))
plt.plot(SNR, np.log10(synth_error), linewidth=3)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()
