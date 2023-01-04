import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fftfreq, fftshift
import os
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
import matplotlib as mpl
from scipy import interpolate

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
tableau10 = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f']#, '#bab0ac']
tableauCB = ['#1170aa', '#fc7d0b', '#a3acb9', '#57606c', '#5fa2ce', '#c85200', '#7b848f', '#a3cce9', '#ffbc79', '#c8d0d9']
mpl.rcParams['axes.prop_cycle'] = cycler(color=tableau10)

plt.figure(figsize=(4,1.5))

x = np.linspace(0,2*np.pi,1000)
# plt.plot(np.sin(x), c=tableau10[8])
plt.plot(np.sin(2* x-0.5*np.pi)/2, c=tableau10[3])
# plt.plot(np.sin(x)+0.5*np.sin(2*x-0.5*np.pi), c=tableau10[2])
plt.axhline(0, lw=0.4, c='k')
plt.ylim((-2.1,2.1))
plt.axis('off')
plt.savefig('2.pdf')
# plt.savefig