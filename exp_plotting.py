import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker


#mfp=mfp_range, c=c_range, spec_rad_oci=spec_rad_oci, spec_rad_pacc=spec_rad_pacc
with np.load('numerical_exp.npz') as data:
    mfp_range = data['mfp']
    c_range = data['c']
    spec_rad_oci = data['spec_rad_oci']
    spec_rad_acc = data['spec_rad_pacc']


print()
print("Maximum ρ for SMM: %.3f, for OCI: %.3f"%(np.max(spec_rad_acc), np.max(spec_rad_oci)))
print()

c, mfp = np.meshgrid(c_range, mfp_range)

fig, (axs) = plt.subplots(ncols=3, layout="constrained")
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
surf = ax1.contourf(mfp, c, spec_rad_acc, levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)

surf2 = ax2.contourf(mfp, c, spec_rad_oci, levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)

fig.colorbar(surf2, ax=axs[1], label=r'$\rho_e$', ticks=[0, .2, .4, .6, .8, 1.0])

surf.set_edgecolor("face")
surf2.set_edgecolor("face")

ax1.set_xscale('log')
ax2.set_xscale('log')

ax1.set_xlabel(r"$\delta$")
ax1.set_ylabel(r"$c$")
ax2.set_xlabel(r"$\delta$")

ax1.set_title(r"2$^{nd}$ Moment Method")
ax2.set_title("OCI")
ax3.set_title(r"$\rho$ Improvement")


rho_acc = spec_rad_oci/spec_rad_acc # computing the ratio

print("max ratio speedup: %.3f"%np.max(rho_acc))

rho_acc = np.ma.masked_where(rho_acc <= 0, rho_acc) # masking 0s

surf_acc = ax3.contourf(mfp, c, rho_acc, locator=ticker.LogLocator(subs='auto'), levels=100, cmap=cm.viridis, antialiased=True)
speed = ax3.contour(mfp, c, rho_acc, levels=[1.0], colors=('w',), linewidths=(3,))
ax3.clabel(speed, fmt='%2.1f', colors='w', fontsize=20)
surf_acc.set_edgecolor("face")
ax3.set_xlabel(r"$\delta$")
ax3.set_xscale('log')
fig.colorbar(surf_acc, ax=axs.ravel().tolist(), label=r'$r$', ticks=[10e-1, 10e0, 10e1, 10e2, 10e3])


plt.gcf().set_size_inches(8, 4)
ax2.label_outer()
ax3.label_outer()
plt.savefig("smm_acc.pdf")
plt.show()




fig, (axs) = plt.subplots(layout="constrained") #subs='all' 'auto'
ax1 = axs
surf = ax1.contourf(mfp, c, rho_acc, locator=ticker.LogLocator(subs='auto'), levels=100, cmap=cm.viridis, antialiased=True)
speed = ax1.contour(mfp, c, rho_acc, levels=[1.0], colors=('w',), linewidths=(3,))
ax1.clabel(speed, fmt='%2.1f', colors='w', fontsize=20)
surf.set_edgecolor("face")
ax1.set_xlabel(r"$\delta$")
ax1.set_ylabel(r"$c$")
ax1.set_xscale('log')

fig.colorbar(surf, label=r'r', ticks=[10e-1, 10e0, 10e1, 10e2, 10e3])
plt.savefig("rho_ratio.pdf")
plt.show()