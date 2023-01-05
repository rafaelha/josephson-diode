#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
from cycler import cycler

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
tableau10 = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f']#, '#bab0ac']
tableauCB = ['#1170aa', '#fc7d0b', '#a3acb9', '#57606c', '#5fa2ce', '#c85200', '#7b848f', '#a3cce9', '#ffbc79', '#c8d0d9']
mpl.rcParams['axes.prop_cycle'] = cycler(color=tableau10)

SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIGGER_SIZE = 15

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

t0 = time.time()

N = 400
rbound = 0.5
N_draws = 1e5
N_r = 10000
N_draws = int(N_draws / N_r)

N_harmonics = 10
harmonics = np.arange(1,N_harmonics+1)

phi = np.linspace(-np.pi,np.pi,N)


harmonics_cos = np.cos(harmonics[None, :] * phi[:, None])
harmonics_sin = np.sin(harmonics[None, :] * phi[:, None])




eta_opt = 0
I_opt = np.zeros(N)
r1_opt = np.zeros(N_harmonics)
r2_opt = np.zeros(N_harmonics)

for i in range(N_draws):
    r1 = rbound * 2 * (np.random.rand(N_harmonics, N_r) - 0)
    r2 = rbound * 2 * (np.random.rand(N_harmonics, N_r) - 0)
    r1[0] = 1
    r2[0] = 0
    r2 = r2 * 0

    I = np.sum(harmonics_cos[:,:,None] * r1[None,:] + harmonics_sin[:,:,None] * r2[None,:], axis=1)
    Imax = np.max(I, axis=0)
    Imin = np.abs(np.min(I, axis=0))

    eta_r = np.abs(Imax - Imin) / (Imax + Imin)
    index_r = np.argmax(eta_r)
    eta = eta_r[index_r]

    if eta > eta_opt:
        eta_opt = eta
        I_opt = I[:,index_r]
        r1_opt = r1[:,index_r]
        r2_opt = r2[:,index_r]


coeffs = (1 - harmonics / (N_harmonics+1) )[None,:]
I_ana = np.sum(coeffs * harmonics_cos, axis=1)
Imax_ana = np.max(I_ana, axis=0)
Imin_ana = np.abs(np.min(I_ana, axis=0))
eta_ana = np.abs(Imax_ana - Imin_ana) / (Imax_ana + Imin_ana)

print('MC: ', eta_opt)
print('analytical: ', eta_ana)
print('analytical: ', (N_harmonics-1)/(N_harmonics+1))
plt.figure(figsize=(10,4))
plt.clf()
plt.subplot(121)
def nm(x):
    return x / np.max(x)
plt.plot(phi/2/np.pi,nm(I_opt), label='Monte Carlo')
plt.plot(phi/2/np.pi,nm(I_ana), '--', label='Analytical')
plt.xlabel('$\\varphi/2\pi$')
plt.ylabel('I')
plt.title(f'N={N_harmonics}')

plt.legend()


plt.subplot(122)
plt.plot(np.abs(r1_opt), '.-')
plt.plot(np.abs(r2_opt), '.-')

plt.axhline(0.5, alpha=0.3)

print('elapsed time (s): ',np.round(time.time() - t0, 2))


#%%
plt.rcParams["text.usetex"] = True

etas = []

N = 20000
N_branches = 100
nn = np.arange(1,N_branches)
subset = [1,2,5,20]
for N_harmonics in nn:
    # N_harmonics = 100

    harmonics = np.arange(1,N_harmonics+1)

    phi = np.linspace(-np.pi,np.pi,N)

    harmonics_cos = np.cos(harmonics[None, :] * phi[:, None])
    harmonics_sin = np.sin(harmonics[None, :] * phi[:, None])

    coeffs = (1 - harmonics / (N_harmonics + 1) )[None,:]
    coeffs = ((N_harmonics+ 1 - harmonics) / (N_harmonics) )[None,:]
    # coeffs = (1 / np.sqrt(harmonics) )[None,:]
    I = np.sum(coeffs * harmonics_cos, axis=1)
    Imax = np.max(I, axis=0)
    Imin = np.abs(np.min(I, axis=0))
    eta = np.abs(Imax - Imin) / (Imax + Imin)

    # print(eta)
    etas.append(eta)

    plt.figure('func', figsize=(5.6,3.4))
    if N_harmonics in subset:
        plt.subplot(121)
        plt.plot((phi+np.pi/2)/2/np.pi, I / ((N_harmonics+1)/2), label=f'$N={N_harmonics}$')
        plt.xlabel('$\\varphi_1/2\pi$')
        plt.ylabel('$2I/[I_0 (N+1)]$')


# plt.legend()
plt.figure('func')
plt.subplot(122)
plt.plot(nn, (nn-1)/(nn+1), 'o--', markersize=1, c='orange', label='$(N-1)/(N+1)$')
plt.plot(nn, etas, 'k.')
plt.ylabel('Diode efficiency $\eta$')
plt.xlabel('$N$')
plt.axhline(1, alpha=0.4, lw=0.6, c='k')
plt.ylim((0,1))
plt.legend()
plt.xlim((1,N_branches))
plt.tight_layout()
# plt.savefig('gen-snail-efficiency.pdf')

plt.figure()
plt.clf()
plt.plot(phi,I)
ind_min = np.argmin(I)
plt.axvline(phi[ind_min])
plt.title(f'{phi[ind_min]/np.pi} - val: {I[ind_min]}')


#%%
from scipy.special import factorial

plt.figure('factorial')
plt.clf()

N_ = 5

NN = np.arange(1,N_+1)

plt.semilogy(factorial(NN)/NN, '.')



