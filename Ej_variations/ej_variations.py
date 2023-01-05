import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from cycler import cycler
from scipy import interpolate


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

#%%

di = 1 +0.05
phi = np.linspace(-4*np.pi,4*np.pi,1000)
phi = np.linspace(-np.pi,np.pi,1000)
phi2 = np.arcsin( (1 - di) * np.sin(phi))
phi2_ = np.pi - np.arcsin( (1 - di) * np.sin(phi))
from scipy import signal


# square = (signal.square(phi+0.5*np.pi, duty=0.5) + 1)/2
# plt.plot(phi/np.pi, square)

# plt.plot(phi/np.pi,phi2*square)
# plt.plot(phi/np.pi,phi2_*(1-square))
# func = phi2*square + phi2_*(1-square) +  z - (z % (2*np.pi))

def get_phi(phi, di):
    phi2 = np.arcsin( di * np.sin(phi))
    phi2_ = np.pi - np.arcsin( di * np.sin(phi))
    z = (phi + 0.5*np.pi)
    square = (signal.square(phi+0.5*np.pi, duty=0.5) + 1)/2
    return phi2*square + phi2_*(1-square) +  z - (z % (2*np.pi))


plt.plot(phi/np.pi, get_phi(phi, di)/np.pi)

plt.axhline(0,lw=0.3)
plt.axvline(0,lw=0.3)

plt.xlim((-1,1))
plt.ylim((-1,1))
plt.axis('equal')

#%%
ax = np.newaxis

N = 5
n = np.arange(1,N+1)
nk = math.factorial(N) / n
n1 = int(nk[0])

plot = False


Ic = 1 - n / (N + 1)
phi_ext = (n - 1) * np.pi / 2
phi_ext_n = phi_ext * nk

variance = 0.1


rand = []
dis = []
for i,ni in enumerate(nk):
    rand.append(np.sort(variance*2*(np.random.rand(int(ni)) - 0.5)))
    dis.append((Ic[i]+rand[i][0])/(Ic[i]+rand[i]))

phi = np.linspace(0,2*np.pi,100)





centre = 0.5 * np.pi
w = 0.1
phases = np.linspace(centre - w, centre + w, 10)
phases = np.linspace(0, 2*np.pi, 300)
CURRENT = []

for x in phases:
    phis0 = get_phi(x, dis[0])
    # plt.figure()
    # plt.plot(phis0, '.')

    current = np.zeros(N) # current in each arm
    current[0] = (Ic[0] + rand[0][0]) * np.sin(x)


    for arm in range(1, N):
        n1phi1 = n1*x - phi_ext_n[arm]
        xc = n1phi1 / nk[arm]
        r = 0.25*N
        xx = np.linspace(xc - r*np.pi, xc + r*np.pi,400)
        sum_phik = np.sum(get_phi(xx[:,ax], dis[arm][ax,:]),axis=1)

        sum_phik_flux = np.sum(phis0) - phi_ext_n[arm]

        f = interpolate.interp1d(sum_phik, xx)
        phi_arm = f(sum_phik_flux)

        if plot:
            plt.figure()
            plt.plot(xx,sum_phik, '.')

            plt.axhline(sum_phik_flux)
            plt.axhline(n1phi1, c='r', ls='--')
            plt.axvline(phi_arm)

        current[arm] = (Ic[arm] + rand[arm][0]) * np.sin(n1/nk[arm]*x - phi_ext[arm])

    current_all = np.sum(current)
    CURRENT.append(current_all)




plt.figure()
plt.plot(phi/np.pi, np.sum(Ic[:,ax] * np.sin(n1/nk[:,ax] * phi[ax,:] - phi_ext[:,ax]), axis=0))
# plt.plot([x/np.pi], current_all, '.')
plt.plot(phases/np.pi, CURRENT, '-')

def eff(maxc, minc):
    return (maxc - np.abs(minc)) / (maxc + np.abs(minc))
maxC = np.max(CURRENT)
minC = np.min(CURRENT)
eff2 = eff(maxC, minC)
plt.title(f'eff_theory={(N-1)/(N+1)}, eff={eff2}')