# coding: utf-8
import numpy as np
a = np.load("/d1/checM/Paris_March2017/Run06992_dl1.npz")
m = a['baseline_rms_full'].mean(1)
n = [m[i*10:(i+1)*10].mean() for i in range(int(len(m)/10))]
plt.plot(n)
import matplotlib.pyplot as plt
plt.plot(n)
plt.show()
a.keys()
a['sec']
len(a['sec'])
a['ns']
get_ipython().magic('pinfo np.save')
get_ipython().magic('pinfo np.save')
np.save('a.variance',(a['sec'],a['ns'],a['baseline_rms_full'].mean(1)))
np.savetxt('a.var',(a['sec'],a['ns'],a['baseline_rms_full'].mean(1),a['baseline_rms_full'].std(1)),comments = 'seconds | nanoseconds | baseline RMS mean | baseline RMS mean error')
np.savetxt('a.var',((a['sec'],a['ns'],a['baseline_rms_full'].mean(1),a['baseline_rms_full'].std(1))),comments = 'seconds | nanoseconds | baseline RMS mean | baseline RMS mean error')
np.savetxt('a.var',np.transpose([a['sec'],a['ns'],a['baseline_rms_full'].mean(1),a['baseline_rms_full'].std(1)]),comments = 'seconds | nanoseconds | baseline RMS mean | baseline RMS mean error')
np.savetxt('a.var',np.transpose([a['sec'],a['ns'],a['baseline_rms_full'].mean(1),a['baseline_rms_full'].std(1)]),header = '# seconds | nanoseconds | baseline RMS mean | baseline RMS mean error')
np.savetxt('a.var',np.transpose([a['sec'],a['ns'],a['baseline_rms_full'].mean(1),a['baseline_rms_full'].std(1)]),header = 'seconds | nanoseconds | baseline RMS mean | baseline RMS mean error')
import readline
readline.write_history_file('a')
get_ipython().magic('save a 1-23')
