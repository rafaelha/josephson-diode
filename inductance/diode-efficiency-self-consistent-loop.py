#%%
import numpy as np
import matplotlib.pyplot as plt
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

plt.rcParams["text.usetex"] = True
# plt.rcParams["text.usetex"] = True

nk = np.array([2, 1])
# nk = np.array([1,1])
# nk = np.array([6, 3, 2])
# nk = np.array([12, 6, 4, 3])
n1 = nk[0]
N = len(nk)
n = np.arange(1,N+1)
N_phi = 500
phi_ = np.linspace(0,2*np.pi,N_phi)

beta = np.ones(N) * 3.9
# beta = np.array([1.5, 0])

Ic = 1 - n / (N + 1)
# Ic = np.array([1,1])
phi_ext = (1 - n) * np.pi / 2


def self_consistent_current(I_ini, phi_ext, Ic):
    I = I_ini
    I_new = I*0 + 1
    # for i in range(50):
    count = 0
    while np.sum(np.abs(I - I_new)) > 1e-10 and count < 5000:
        shift = phi_ext[:,None] + beta[0] * I[0] / n1 - beta[:,None] * I / nk[:,None]
        phi = n1 / nk[:,None] * phi_[None,:] + shift
        I_new = Ic[:,None] * np.sin(phi)
        alpha = 0.6
        I = alpha * I_new + (1-alpha) * I
        count += 1
    print(count)

    Isum = np.sum(I, axis=0)
    Imax = np.max(Isum)
    Imin = -np.min(Isum)

    shift_ind_max = np.argmax(Isum)
    shift_ind_min = np.argmin(Isum)

    return Imax, Imin, I, shift[:,shift_ind_max], shift[:,shift_ind_min]

# eta = np.abs(Imax-Imin) / (Imax + Imin)
# print(eta)

Phis_max = []
Phis_min = []
Imaxs = []
Imins = []
etas = []

NPhis = 100
x = np.concatenate([np.linspace(0,2*np.pi,NPhis), np.linspace(2*np.pi, 0,NPhis)])


I = np.zeros((N,N_phi))
for p in x:
    phi_ext = np.array([0,p])
    Imax, Imin, I, Phi_max, Phi_min = self_consistent_current(I, phi_ext, Ic)
    Phis_max.append(Phi_max[1])
    Phis_min.append(Phi_min[1])
    etas.append(np.abs(Imax - Imin)/(Imax + Imin))
    Imaxs.append(Imax)
    Imins.append(Imin)




plt.figure('quantization')
plt.clf()
plt.subplot(411)
plt.plot(x/(2*np.pi), Imaxs, '-')
plt.plot(x/(2*np.pi), -np.array(Imins))

plt.subplot(412)
plt.plot(x[:NPhis]/(2*np.pi), np.array(Phis_max)[:NPhis]/(2*np.pi), '.', c='blue', markersize=0.3)
plt.plot(x[NPhis:]/(2*np.pi), np.array(Phis_max)[NPhis:]/(2*np.pi), '.', c='orange', markersize=0.3)
# plt.plot(x[:NPhis]/(2*np.pi), np.array(Phis_max)[:NPhis]/(2*np.pi), 'blue', ls='-')
# plt.plot(x[NPhis:]/(2*np.pi), np.array(Phis_max)[NPhis:]/(2*np.pi), 'orange', ls='-')
plt.axvline(0.25, lw=0.3, ls='--')
plt.axvline(0.5, lw=0.3)
plt.axvline(0.75, lw=0.3, ls='--')
plt.axvline(1.5, lw=0.3)
plt.axvline(2.5, lw=0.3)
plt.axhline(0, lw=0.3)
plt.axhline(1, lw=0.3)
plt.axhline(2, lw=0.3)
plt.axhline(3, lw=0.3)

plt.subplot(413)
plt.plot(x[:NPhis]/(2*np.pi), np.array(Phis_min)[:NPhis]/(2*np.pi), '.', c='green', markersize=0.3)
plt.plot(x[NPhis:]/(2*np.pi), np.array(Phis_min)[NPhis:]/(2*np.pi), '.', c='black', markersize=0.3)
# plt.plot(x[:NPhis]/(2*np.pi), np.array(Phis_min)[:NPhis]/(2*np.pi), 'blue', ls='--')
# plt.plot(x[NPhis:]/(2*np.pi), np.array(Phis_min)[NPhis:]/(2*np.pi), 'orange', ls='--')
plt.axvline(0.25, lw=0.3, ls='--')
plt.axvline(0.5, lw=0.3)
plt.axvline(0.75, lw=0.3, ls='--')
plt.axvline(1.5, lw=0.3)
plt.axvline(2.5, lw=0.3)
plt.axhline(0, lw=0.3)
plt.axhline(1, lw=0.3)
plt.axhline(2, lw=0.3)
plt.axhline(3, lw=0.3)

plt.subplot(414)
plt.plot(x[:NPhis]/(2*np.pi), np.array(etas)[:NPhis])
plt.plot(x[NPhis:]/(2*np.pi), np.array(etas)[NPhis:])
plt.title(np.max(etas))

plt.tight_layout()

#%%
import scipy.special
import math

# nk = np.array([12, 6, 4, 3])
N = 20
n = np.arange(1,N+1)
nk = math.factorial(N) / n
n1 = nk[0]


beta = np.ones(N) * 2 * nk

Ic = 1 - n / (N + 1)
phi_ext = (1 - n) * np.pi / 2

Imax, Imin, I, Phi_max, Phi_min = self_consistent_current(np.zeros((N,N_phi)), phi_ext, Ic)

plt.figure('test')
plt.clf()
plt.plot(phi_, np.sum(I, axis=0))
# plt.plot(phi_, I[1])
plt.title(f'$\eta={np.abs(Imax-Imin)/(Imax+Imin)}$')


plt.figure('eff', figsize=(5.6,3.4))
plt.clf()
plt.subplot(121)

bfrange = np.arange(5)/10
for bf in bfrange:
    etas = []
    N = 5
    n = np.arange(1,N+1)
    nk = math.factorial(N) / n
    n1 = nk[0]


    beta = np.ones(N) * bf * n1

    Ic = 1 - n / (N + 1)
    Ic = (N+1-n) / N
    phi_ext = (1 - n) * np.pi / 2

    Imax, Imin, I, Phi_max, Phi_min = self_consistent_current(np.zeros((N,N_phi)), phi_ext, Ic)

    eta=np.abs(Imax-Imin)/(Imax+Imin)

    etas.append(eta)

    plt.plot(phi_/np.pi/2,np.sum(I, axis=0), label=f'${bf}$')
    # plt.legend()

plt.xlabel('$\\varphi_1/2\pi$')
plt.ylabel('$I/I_0$')
plt.axhline(0,c='darkblue', lw=0.4)


plt.subplot(122)
Ns = np.arange(1,21)
for bf in bfrange:
    etas = []
    for N in Ns:
        n = np.arange(1,N+1)
        nk = math.factorial(N) / math.factorial(int(N/2)) / n
        n1 = nk[0]


        beta = np.ones(N) * bf * n1

        Ic = 1 - n / (N + 1)
        Ic = (N+1-n) / N
        phi_ext = (1 - n) * np.pi / 2

        Imax, Imin, I, Phi_max, Phi_min = self_consistent_current(np.zeros((N,N_phi)), phi_ext, Ic)

        eta=np.abs(Imax-Imin)/(Imax+Imin)

        etas.append(eta)

    plt.plot(Ns,etas, '.-', label=f'${bf}$')

plt.ylabel('Diode efficiency $\eta$')
plt.xlabel('$N$')
plt.ylim((0,1))
plt.xlim((1,20))
# plt.legend()
plt.tight_layout()

plt.savefig('inductance-efficiency-2.pdf')

# #%%

# Phis = []

# PhiEs = np.concatenate([np.linspace(0,6,100), np.linspace(6,0,100)])
# betaE = 9.3

# def getPhi(PhiE, Phi_ini):
#     Phi = Phi_ini
#     alpha = 0.4
#     for i in range(500):
#         Phi = alpha * (PhiE - betaE / (2 * np.pi) * np.sin(2*np.pi*Phi)) + (1-alpha) * Phi

#     return Phi

# Phi_ini = 0
# for PhiE in PhiEs:
#     Phi = getPhi(PhiE, Phi_ini)
#     Phis.append(Phi)
#     Phi_ini = Phi

# plt.figure('flux quantization singel junction')
# plt.clf()
# plt.plot(PhiEs, Phis)