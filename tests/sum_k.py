import numpy as np
import matplotlib.pyplot as plt

a = 10
a_sum = 1000

actual_ll= []
appx_lcb_ll = []
appx_ucb_ll=[]
for a in range(1,100):
    a_ll = 1/np.arange(a,a_sum)
    actual_ll.append(np.sum(a_ll))
    appx_lcb_ll.append(np.log((a_sum)/a))
    appx_ucb_ll.append(np.log((a_sum-1) / a))


plt.plot(np.arange(len(actual_ll))+1, actual_ll)
plt.plot(np.arange(len(actual_ll))+1, appx_lcb_ll)
plt.plot(np.arange(len(actual_ll))+1, appx_ucb_ll)
plt.plot(np.arange(len(actual_ll))+1, 0.5*np.array(appx_ucb_ll) + 0.5*np.array(appx_lcb_ll))
plt.legend(['actual','approx_lower','approx_upper', 'approx_mean'])
plt.show()