import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from cycler import cycler
from scipy import interpolate
from scipy import signal
from scipy.optimize import minimize

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
                  '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
tableau10 = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
             '#edc948', '#b07aa1', '#ff9da7', '#9c755f']  # , '#bab0ac']
tableauCB = ['#1170aa', '#fc7d0b', '#a3acb9', '#57606c',
             '#5fa2ce', '#c85200', '#7b848f', '#a3cce9', '#ffbc79', '#c8d0d9']
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

plt.rcParams["text.usetex"] = True

# %%


# %%
ax = np.newaxis

N = 4
n = np.arange(1, N+1)
nk = math.factorial(N) // n
gcd = np.gcd.reduce(nk)
nk = nk/gcd
n1 = int(nk[0])

Ic = 1 - n / (N + 1)
phi_ext = (n - 1) * np.pi / 2
phi_ext_n = phi_ext * nk



def gen_rand(stdev, seed=None):
    if seed != None:
        np.random.seed(seed)
    rand = []
    dis = []
    for i, ni in enumerate(nk):
        rand_num = np.sort(stdev*np.random.standard_normal(int(ni)))
        if i >= 1:
            pass
            rand_num = np.flip(rand_num)
        rand.append(rand_num)
        dis.append((Ic[i]+rand[i][0])/(Ic[i]+rand[i]))
    return rand, dis





def arcsin(x):
    x[x > 1] = 1
    x[x < -1] = -1
    return np.arcsin(x)
    # return np.nan_to_num(np.arcsin(x), nan=-np.arcsin(1))


def get_phi(phi, di):
    phi2 = arcsin(di * np.sin(phi))
    phi2_ = np.pi - arcsin(di * np.sin(phi))
    z = (phi + 0.5*np.pi)
    square = (signal.square(phi+0.5*np.pi, duty=0.5) + 1)/2
    return phi2*square + phi2_*(1-square) + z - (z % (2*np.pi))


def getI(phi1):
    phis0 = get_phi(phi1, dis[0])  # phases of first arm
    # plt.figure()
    # plt.plot(phis0, '.')

    current = np.zeros(N)  # current in each arm
    current[0] = (Ic[0] + rand[0][0]) * np.sin(phi1)

    valid = 1

    for arm in range(1, N):
        n1phi1 = n1*phi1 - phi_ext_n[arm]
        xc = n1phi1 / nk[arm]
        # r = 0.15*N
        # xx = np.linspace(xc - r*np.pi, xc + r*np.pi,2000)
        # sum_phik = np.sum(get_phi(xx[:,ax], dis[arm][ax,:]),axis=1)

        sum_phik_flux = np.sum(phis0) - phi_ext_n[arm]

        def cost(x):
            sum_phik = np.sum(get_phi(x, dis[arm]))
            return (sum_phik - sum_phik_flux)**2

        # f = interpolate.interp1d(sum_phik, xx)
        # phi_arm = f(sum_phik_flux)

        res = minimize(cost, xc)
        phi_arm = res.x
        current[arm] = (Ic[arm] + rand[arm][0]) * np.sin(phi_arm)
    return np.sum(current)


def eff(current):
    maxc = np.max(current)
    minc = np.min(current)
    return (maxc - np.abs(minc)) / (maxc + np.abs(minc))

phases = np.linspace(0, 2*np.pi, 500)
stdev = 0.005
rand, dis = gen_rand(stdev=stdev, seed=None)
phi = np.linspace(0, 2*np.pi, 100)
CURRENT = [getI(x) for x in phases]

plt.figure()
plt.plot(phi/np.pi, np.sum(Ic[:, ax] * np.sin(n1 / nk[:, ax] * phi[ax, :] - phi_ext[:, ax]), axis=0))
plt.plot(phases/np.pi, CURRENT, '-')
plt.title(f'eff_theory={(N-1)/(N+1)}, eff={np.round(eff(CURRENT),4)}')


#%%
disorder_averages = 1
Nmax = 10
stdevs = [0, 0.001, 0.005, 0.01, 0.02]

EFF = np.zeros((disorder_averages,Nmax, len(stdevs)))

import time

start = time.time()

for Ni in range(1,Nmax):
    for seed in range(disorder_averages):
        for vi,stdev in enumerate(stdevs):
            N = Ni + 1
            n = np.arange(1, N+1)
            nk = math.factorial(N) // n
            gcd = np.gcd.reduce(nk)
            nk = nk/gcd
            n1 = int(nk[0])

            Ic = 1 - n / (N + 1)
            phi_ext = (n - 1) * np.pi / 2
            phi_ext_n = phi_ext * nk

            phases = np.linspace(0, 2*np.pi, 500)
            rand, dis = gen_rand(stdev=stdev, seed=seed)

            CURRENT = [getI(x) for x in phases]
            EFF[seed, Ni, vi] = eff(CURRENT)


Ns = np.arange(0,N) + 1
EFF[:,:,0] = ((Ns-1)/(Ns+1))[ax,:]

print('elapsed time: ', time.time() - start)

#%%
EFFavg = np.mean(EFF, axis=0)
plt.figure('eff', figsize=(5.6,3.4))
plt.clf()
plt.plot(np.arange(N)+1, EFFavg, '.-')
plt.legend(stdevs)
plt.ylabel('Diode efficiency $\eta$')
plt.xlabel('$N$')
plt.legend([f'$\sigma={sigma*100}\%$' for sigma in stdevs])


# %%
# phi = np.linspace(-3*np.pi, 3*np.pi, 1000)
# di = 0.05
# plt.plot(phi/np.pi, get_phi(phi, 1-di)/np.pi)
# plt.plot(phi/np.pi, get_phi(phi, 1+di)/np.pi)

# plt.axhline(0, lw=0.3)
# plt.axvline(0, lw=0.3)

# plt.xlim((-1, 1))
# plt.ylim((-1, 1))
# plt.axis('equal')
