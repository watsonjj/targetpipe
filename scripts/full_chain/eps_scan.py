import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

hillas_parameters = np.loadtxt('hillas_parameters_test.dat',delimiter=',', unpack=True)
cenx = []
ceny = []
width = []
length = []
psi = []

for i in range(len(hillas_parameters[0])):
    if math.isnan(hillas_parameters[4][i]) or math.isnan(hillas_parameters[5][i]):
        continue
    elif hillas_parameters[4][i] ==0.0 or hillas_parameters[5][i] ==0.0:
        continue
    else:
        cenx.append(hillas_parameters[2][i])
        ceny.append(hillas_parameters[3][i])
        width.append(hillas_parameters[5][i])
        length.append(hillas_parameters[4][i])
        psi.append(hillas_parameters[8][i])

cenx = np.asarray(cenx)
ceny = np.asarray(ceny)
width = np.asarray(width)
length = np.asarray(length)
psi = np.asarray(psi)


eps = np.arange(0,0.3,0.001)
# eps = np.arange(1,3,0.01)
fig3 = plt.figure(3, figsize=[7,7])
# ax3 = fig3.add_subplot(111)
ax1 = fig3.add_subplot(221)
ax2 = fig3.add_subplot(222)
ax3 = fig3.add_subplot(223)
ax4 = fig3.add_subplot(224)

plt.ion()
x80=[]
eps80=[]

for n, ep in enumerate(eps):
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    ax4.set_xlim(eps[0], eps[-1])
    rx = cenx-ep * (1 - width / length) * np.cos(psi)
    ry = ceny-ep * (1 - width / length) * np.sin(psi)
    miss = np.sqrt(rx**2+ry**2)
    h1, x1, im1 = ax3.hist(miss, bins=100)
    sum1 = 0
    idx=0
    for i in range(len(h1)):
        if sum1/float(sum(h1))>0.80:
            break
        sum1 = sum1+h1[i]
        idx += 1
    ax3.plot([x1[idx],x1[idx]],[0,h1[idx]],color = 'k', ls =':')
    x80.append(x1[idx])
    eps80.append(ep)
    ax4.plot(eps80,x80,color='k')
    h ,x, y, im =ax1.hist2d(rx, ry, bins=np.arange(-0.15, 0.151, 0.005))
    ax1.text(-0.1,0.13,r'$\epsilon=$%s' % ep, color='w')
    ax1.set_xlabel('x camera position [m]')
    ax1.set_ylabel('y camera position [m]')
    ax2.plot(y[:-1], h[30])
    ax2.plot(x[:-1], h[29])
    ax2.plot(x[:-1], h[31])
    ax2.plot(x[:-1], h[32])
    ax2.plot(x[:-1], h[28])

    ax2.set_xlabel('x camera position [m]')
    ax3.set_xlabel('dist from 0')
    ax4.set_xlabel(r'$\epsilon$')
    ax4.set_ylabel(r'$R_{80}$')
    if n == len(eps):
        plt.show()
    else:
        plt.pause(0.0001)
