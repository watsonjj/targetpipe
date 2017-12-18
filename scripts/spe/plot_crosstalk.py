from matplotlib import pyplot as plt, animation
import numpy as np
from scipy.special import binom
from scipy.stats import poisson
from IPython import embed


def poct(opct, j, n):
    return np.power(1-opct, j) * np.power(opct, n - j) * binom(n-1, j-1)


def pj(j, lambda_):
    return poisson.pmf(j, lambda_)


j = np.linspace(0, 20, 20+1)
init = np.ma.masked_array(j, mask=True)
p = pj(j, 2)

fig1, ax1 = plt.subplots()
l1, = ax1.plot(j, init, label="0.3")
l2, = ax1.plot(j, init, label="0.5")
l3, = ax1.plot(j, init, label="0.8")
l4, = ax1.plot(j, init, label="Probability of j initial cells fired")
ax1.set_xlabel("j")
ax1.set_ylabel("Poct")
ax1.legend()

fig2, ax2 = plt.subplots()
l5, = ax2.plot(j, init, label="0.3")
l6, = ax2.plot(j, init, label="0.5")
l7, = ax2.plot(j, init, label="0.8")
ax2.set_xlabel("j")
ax2.set_ylabel("Poct")
ax2.legend()


def anim_oct(i):
    n = i
    y1 = poct(0.3, j, n)
    y2 = poct(0.5, j, n)
    y3 = poct(0.8, j, n)

    # print(np.sum(y1))
    # print(np.sum(y2))
    # print(np.sum(y3))

    l1.set_ydata(y1)
    l2.set_ydata(y2)
    l3.set_ydata(y3)
    l4.set_ydata(p)

    ax1.relim()
    ax1.autoscale_view()

    ax1.set_title("Probability to have n={} cells fired from crosstalk".format(n))


def anim_total(i):
    n = i
    y1 = poct(0.3, j, n) * p
    y2 = poct(0.5, j, n) * p
    y3 = poct(0.8, j, n) * p

    # print(np.sum(y1))
    # print(np.sum(y2))
    # print(np.sum(y3))

    l5.set_ydata(y1)
    l6.set_ydata(y2)
    l7.set_ydata(y3)

    ax2.relim()
    ax2.autoscale_view()

    ax2.set_title("Probability to have n={} fired cells total".format(n))


ani1 = animation.FuncAnimation(fig1, anim_oct, np.arange(0, 20), interval=250)
ani2 = animation.FuncAnimation(fig2, anim_total, np.arange(0, 20), interval=250)


def prob_jason(opct, j):
    n = np.arange(10)
    p_total = np.sum(p[:, None] * poct(opct, j[:, None], n[None, :]), 0)
    return p_total

def prob_rich(opct):
    j = 21
    n = 10

    p0 = np.zeros(j)
    for ji in range(0, j):
        p0[ji] = pj(ji, 2)

    p = np.zeros(n)
    p[0] = p0[0]
    for ni in range(1, n):
        for ji in range(1, j):
            p[ni] += p0[ji] * (1 - opct) ** ji * opct ** (ni - ji) * binom(ni - 1, ji - 1)

    return p

x = np.arange(10)
y1 = prob_jason(0.3, j)
y2 = prob_jason(0.5, j)
y3 = prob_jason(0.8, j)
y4 = prob_rich(0.3)
y5 = prob_rich(0.5)
y6 = prob_rich(0.8)


fig3, ax3 = plt.subplots()
ax3.plot(x, y1, label="j 0.3")
ax3.plot(x, y2, label="j 0.5")
ax3.plot(x, y3, label="j 0.8")
ax3.plot(x, y4, label="r 0.3")
ax3.plot(x, y5, label="r 0.5")
ax3.plot(x, y6, label="r 0.8")
ax3.set_xlabel("n")
ax3.set_ylabel("P total")
ax3.legend()


plt.show()