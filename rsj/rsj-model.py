#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams["text.usetex"] = True

# alpha = phi'' + beta phi' + sin phio

# y1 = phi
# y2 = phi'

# y1' = y2
# y2' = phi'' = alpha - beta * y2 - sin y1

def get_n1(N):
    ks=np.arange(1,N+1)
    nks=np.math.factorial(N)/(ks)/np.math.factorial(int(N/2))
    nks=nks.astype(int)
    gcd=np.gcd.reduce(nks)

    return nks[0]/gcd

# get_n1(10)
N = 8 # number of harmonics
n1 = get_n1(N)
beta =  2.6 * np.sqrt(n1) # conductance
beta = 5
n = np.arange(1, N+1)


#%%


def current(phase):
    # return np.sin(phase)
    Ic = 1 - n / (N + 1)
    Ic = (N+1-n) / N
    phi_ext = -(1 - n) * np.pi / 2
    if hasattr(phase, '__len__'):
        return np.sum(Ic[:,None] * np.sin(n[:,None]*phase[None,:] - phi_ext[:,None]), axis=0)
    else:
        # Ic = 1 - n / (N + 1)
        # Ic = (N+1-n) / N
        # phi_ext = (1 - n) * np.pi / 2
        return np.sum(Ic * np.sin(n*phase - phi_ext))

phi_ = np.linspace(0,2*np.pi,int(1e3))
phi0 = phi_[np.argmin(current(phi_))]


def dy(t, y):
    return np.array([y[1], alpha - beta * y[1] + current(y[0])])

imax = 0.8 * (N+1)
imin = -imax
N_i = 250
N_i = 50

current_path = np.concatenate([np.linspace(0,imax,N_i), np.linspace(imax,imin,2*N_i), np.linspace(imin,0,N_i)])
tmax = 400
Nt = 500
t_span = np.linspace(0,tmax,Nt)


y1 = phi0
y2 = 0

v = []
def extract_v(t,y):
    # return (y[-1] - y[len(y)//2]) / (t[-1] - t[len(t)//2])
    p = (Nt//2)
    m,b = np.polyfit(t[p:], y[p:], 1)
    # plt.figure()
    # plt.plot(t,y)
    # plt.plot(t, t*m + b)
    return m

for alpha in current_path:
    sol = solve_ivp(dy, (0, tmax), t_eval=t_span, y0=[y1,y2], method='RK23', rtol=1e-3, atol=1e-6)

    v_ = extract_v(sol.t, sol.y[0])
    v.append(v_)

    y1 = sol.y[0,-1] % (2*np.pi)
    y2 = sol.y[1,-1]
    # y1 = 0
    # y2 = v_

# plt.figure()
# plt.plot(sol.t, sol.y[0])

#%%
plt.figure('a', figsize=(2.5,2.5))
plt.clf()
plt.plot(np.array(v)*beta/(N+1), current_path/(N+1), c='k', lw=1)
plt.axhline(0, lw=0.3)
plt.xlabel('$V/(n_1RI_0(N+1))$')
plt.ylabel('$I_{dc}/[I_0(N+1)]$')
# plt.title(f'$N={N}$')
plt.tight_layout()
plt.text(-0.75,0.69,f'$N={N}$', size=13)
# plt.savefig(f'rsj-{N}.pdf')

plt.figure('a', figsize=(2,2))
plt.clf()
plt.plot(np.array(v)*beta/(N+1)*2, current_path/(N+1)*2, c='k', lw=1)
plt.axhline(0, lw=0.7, c='g')
plt.xlim((-1.6,1.6))
plt.ylim((-1.6,1.6))
plt.xlabel('$V$')
plt.ylabel('$I$')
# plt.title(f'$N={N}$')
plt.tight_layout()
# plt.text(-0.75,0.69,f'$N={N}$', size=13)
plt.savefig(f'diode-{N}-trs.png', dpi=200)


#%%
plt.figure('b', figsize=(2.5,2.5))
plt.clf()
x = np.linspace(-1.6,1.6,400)
y = np.exp(x/0.2)-1
plt.plot(x,y/30,c='k', lw=1)
plt.axhline(0, lw=0.7, c='g')
plt.axvline(0, lw=0.7, c='g')
plt.xlim((-1.6,1.6))
plt.ylim((-0.3,1.1))
plt.xlabel('$V$')
plt.ylabel('$I$')
# plt.title(f'$N={N}$')
plt.tight_layout()
# plt.text(-0.75,0.69,f'$N={N}$', size=13)
# plt.savefig(f'diode.png', dpi=200)