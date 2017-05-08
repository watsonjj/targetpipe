# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

#laser run
a = np.load("/d1/checM/Paris_March2017/Run07335_dl1_a.npz")

b = a['baseline_rms_full']

c = np.sqrt((b**2).sum(0))

d = b.mean(0)

e = a['baseline_mean_full']
f = e.sum(0)



a1 = np.load("/d1/checM/Paris_March2017/Run07337_dl1_a.npz")

b1 = a1['baseline_rms_full']

c1 = np.sqrt((b1**2).sum(0))

d1 = b1.mean(0)

e1 = a1['baseline_mean_full']
f1 = e1.sum(0)



#Initial lid closed, HV on
a2 = np.load("/d1/checM/Paris_March2017/Run07338_dl1.npz")
b2 = a2['baseline_rms_full']
c2 = np.sqrt((b2**2).sum(0))

d2 = b2.sum(0)

e2 = a2['baseline_mean_full']
f2 = e2.sum(0)


fig, axarr = plt.subplots(2)

axarr[0].plot(c-c1,label='uu')
axarr[1].plot(d-d1)



#laser run
a = np.load("/d1/checM/Paris_March2017/Run07340_dl1.npz")

b = a['baseline_rms_full']

c = np.sqrt((b**2).sum(0))

d = b.mean(0)

e = a['baseline_mean_full']
f = e.sum(0)



a1 = np.load("/d1/checM/Paris_March2017/Run07341_dl1.npz")

b1 = a1['baseline_rms_full']

c1 = np.sqrt((b1**2).sum(0))

d1 = b1.mean(0)

e1 = a1['baseline_mean_full']
f1 = e1.sum(0)

axarr[0].plot(c-c1,label='cu')
axarr[1].plot(d-d1)








#laser run
a = np.load("/d1/checM/Paris_March2017/Run07342_dl1.npz")

b = a['baseline_rms_full']

c = np.sqrt((b**2).sum(0))

d = b.mean(0)

e = a['baseline_mean_full']
f = e.sum(0)



a1 = np.load("/d1/checM/Paris_March2017/Run07343_dl1.npz")

b1 = a1['baseline_rms_full']

c1 = np.sqrt((b1**2).sum(0))

d1 = b1.mean(0)

e1 = a1['baseline_mean_full']
f1 = e1.sum(0)


axarr[0].plot(c-c1,label='cc')
axarr[1].plot(d-d1)



#laser run
a = np.load("/d1/checM/Paris_March2017/Run07344_dl1.npz")

b = a['baseline_rms_full']

c = np.sqrt((b**2).sum(0))

d = b.mean(0)

e = a['baseline_mean_full']
f = e.sum(0)



a1 = np.load("/d1/checM/Paris_March2017/Run07345_dl1.npz")

b1 = a1['baseline_rms_full']

c1 = np.sqrt((b1**2).sum(0))

d1 = b1.mean(0)

e1 = a1['baseline_mean_full']
f1 = e1.sum(0)


axarr[0].plot(c-c1,label='uc')
axarr[1].plot(d-d1)





axarr[0].set_title('sqrt(sum(RMS_i**2)) subtracted')
axarr[0].legend()
axarr[0].set_ylabel('p.e.')
axarr[1].set_title('MEAN(RMS_i) subtracted')
axarr[1].set_ylabel('p.e.')
axarr[1].set_xlabel('Pixel ID')
#axarr[2].plot(f)
#axarr[2].set_title('MEAN_i')

plt.show()



