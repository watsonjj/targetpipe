#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from ctapipe.calib import CameraCalibrator
from ctapipe.image import (tailcuts_clean, hillas_parameters)
from ctapipe.visualization import CameraDisplay
from ctapipe.io.hessio import hessio_event_source
import numpy as np
import sys
import math as m

display_each = False
display = False
write = True
eps = 0.08

if display is True:
    figel =plt.figure(10)
    ax = figel.add_subplot(111)
    ax.set_ylim(-0.15,0.15)
    ax.set_xlim(-0.15,0.15)

if __name__ == '__main__':
    calib = CameraCalibrator(None,None)
    size = []
    cen_x = []
    cen_y = []
    length = []
    width = []
    r = []
    phi = []
    psi = []
    miss = []
    skewness = []
    kurtosis = []
    dispp = []
    nbins=100
    dir_x =[]
    dir_y =[]
    disp = None
    dirpath = '/Users/armstrongt/Workspace/CTA/CHEC-S-ASTRI/sicily_analysis/scripts/'
    filelist = np.loadtxt(sys.argv[1], dtype=str)

    # for ep in eps:

    if write is True:
        outfile = open('%shillas_parameters_gamma_mc.dat' % (dirpath), 'w')
        outfile.write('#ID[0],size[1],cenX[2],cenY[3],l[4],w[5],r[6],phi[7],psi[8],miss[9],skewness[10],kurtosis[11], '
                      'recox[12], recoy[13], maxpe [14] || MC: ||, energy[15], alt[16], az[17], core_x[18], core_y[19],'
                      'hfirstint[20\n')

    for filename in filelist:

        source = hessio_event_source(filename, max_events=1000)
        for event in source:
            try:
                calib.calibrate(event)
                if display_each is  True:
                    if disp is None:
                        geom = event.inst.subarray.tel[1].camera
                        disp = CameraDisplay(geom)
                        disp.add_colorbar()
                        plt.show(block=False)
                else:
                    geom = event.inst.subarray.tel[1].camera
                im = event.dl1.tel[1].image[0]
                mask = tailcuts_clean(geom, im , picture_thresh=10, boundary_thresh=5)
                im[~mask] = 0.0
                maxpe = max(im)
                mcenergy = event.mc.energy
                if display_each is True:
                    disp.image = im
                params = hillas_parameters(geom=geom, image=im)

                if params.cen_x.value > 0.2 or params.cen_x.value < -0.2 or params.cen_y.value > 0.2 or params.cen_y.value < -0.2:
                    continue

                if write is True:
                    outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                        event.r0.event_id,
                        params.size,
                        params.cen_x.value,
                        params.cen_y.value,
                        params.length.value,
                        params.width.value,
                        params.r.value,
                        params.phi.value,
                        params.psi.value,
                        params.miss.value,
                        params.skewness,
                        params.kurtosis,
                        params.cen_x.value - eps * (1 - params.width / params.length) * np.cos(params.psi.value),
                        params.cen_y.value - eps * (1 - params.width / params.length) * np.sin(params.psi.value),
                        maxpe,
                        event.mc.energy.value,
                        event.mc.alt.value,
                        event.mc.az.value,
                        event.mc.core_x.value,
                        event.mc.core_y.value,
                        event.mc.h_first_int.value))

                if display==True:
                    size.append(params.size)
                    cen_x.append(params.cen_x.value)
                    cen_y.append(params.cen_y.value)
                    length.append(params.length.value)
                    width.append(params.width.value)
                    r.append(params.r.value)
                    phi.append(params.phi.value)
                    psi.append(params.psi.value)
                    miss.append(params.miss.value)
                    skewness.append(params.skewness)
                    kurtosis.append(params.kurtosis)
                    dispp.append(eps*(1-params.width/params.length))
                    dir_x.append(params.cen_x.value - eps*(1-params.width/params.length)*np.cos(params.psi.value))
                    dir_y.append(params.cen_y.value - eps*(1-params.width/params.length)*np.sin(params.psi.value))
                    ellipse = Ellipse(xy=(params.cen_x.value, params.cen_y.value), width=params.length.value*2,
                                      height=params.width.value*2, angle =np.degrees(params.psi.rad),
                                      edgecolor='r', fc='None', lw=2)
                    ax.add_patch(ellipse)
                    # print(params.cen_x, eps*(1-params.width/params.length)*np.cos(params.phi.rad)+params.cen_x.value)
                    ax.plot([params.cen_x.value,
                             -eps*(1-params.width/params.length)*np.cos(params.psi.rad)+params.cen_x.value],
                            [params.cen_y.value,
                             -eps*(1-params.width/params.length)*np.sin(params.psi.rad)+params.cen_y.value], color='r')
                    if display_each is True:
                        plt.pause(1)
            except Exception as e:
                print(str(e))
                continue
    if display_each is True:
        plt.pause(100)

    if display is True:
        size = [value for value in size if not m.isnan(value)]
        cen_x = [value for value in cen_x if not m.isnan(value)]
        cen_y = [value for value in cen_y if not m.isnan(value)]
        length = [value for value in length if not m.isnan(value)]
        width = [value for value in width if not m.isnan(value)]
        r = [value for value in r if not m.isnan(value)]
        phi = [value for value in phi if not m.isnan(value)]
        psi = [value for value in psi if not m.isnan(value)]
        miss = [value for value in miss if not m.isnan(value)]
        skewness = [value for value in skewness if not m.isnan(value)]
        kurtosis = [value for value in kurtosis if not m.isnan(value)]
        dispp = [value for value in dispp if not m.isnan(value)]
        dir_x = [value for value in dir_x if not m.isnan(value)]
        dir_y = [value for value in dir_y if not m.isnan(value)]

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(3,4,1)
        plt.hist(size, bins=nbins)
        plt.xlabel('size')
        plt.title('size')
        ax2 = fig1.add_subplot(3, 4, 2)
        plt.hist(cen_x, bins=nbins)
        plt.xlabel('cen_x')
        plt.title('cen_x')
        ax3 = fig1.add_subplot(3, 4, 3)
        plt.hist(cen_y, bins=nbins)
        plt.xlabel('cen_y')
        plt.title('cen_y')
        ax4 = fig1.add_subplot(3, 4, 4)
        plt.hist(np.nan_to_num(np.asarray(length)), bins=nbins)
        plt.xlabel('length')
        plt.title('length')
        ax5 = fig1.add_subplot(3, 4, 5)
        plt.hist(np.nan_to_num(np.asarray(width)), bins=nbins)
        plt.xlabel('width')
        plt.title('width')
        ax6 = fig1.add_subplot(3, 4, 6)
        plt.hist(np.nan_to_num(np.asarray(r)), bins=nbins)
        plt.xlabel('r')
        plt.title('r')
        ax7 = fig1.add_subplot(3, 4, 7)
        plt.hist(np.nan_to_num(np.asarray(phi)), bins=nbins)
        plt.xlabel('phi')
        plt.title('phi')
        ax8 = fig1.add_subplot(3, 4, 8)
        plt.hist(np.nan_to_num(np.asarray(psi)), bins=nbins)
        plt.xlabel('psi')
        plt.title('psi')
        ax9 = fig1.add_subplot(3, 4, 9)
        plt.hist(np.nan_to_num(np.asarray(miss)), bins=nbins)
        plt.xlabel('miss')
        plt.title('miss')
        ax10 = fig1.add_subplot(3, 4, 10)
        plt.hist(np.nan_to_num(np.asarray(skewness)), bins=nbins)
        plt.xlabel('skewness')
        plt.title('skewness')
        ax11 = fig1.add_subplot(3, 4, 11)
        plt.hist(np.nan_to_num(np.asarray(kurtosis)), bins=nbins)
        plt.xlabel('kurtosis')
        plt.title('kurtosis')
        ax12 = fig1.add_subplot(3, 4, 12)
        plt.hist(np.nan_to_num(np.asarray(dispp)), bins=nbins)
        plt.xlabel('disp')
        plt.title('disp')
        # fig2 = plt.figure(2)
        # plt.hist2d(cen_x,cen_y, bins=100)
        fig3 = plt.figure(3)
        plt.hist2d(dir_x,dir_y, bins=100)
    if write is True:
        outfile.close()
    if display is True:
        plt.show()