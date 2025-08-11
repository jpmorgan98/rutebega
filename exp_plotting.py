import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

import matplotlib as mpl


#mfp=mfp_range, c=c_range, spec_rad_oci=spec_rad_oci, spec_rad_pacc=spec_rad_pacc
with np.load('numerical_exp.npz') as data:
    mfp_range = data['mfp']
    c_range = data['c']
    spec_rad_oci = data['spec_rad_oci']
    spec_rad_acc = data['spec_rad_pacc']



from matplotlib.colors import ListedColormap

N = 1000
vals = np.ones((N, 4))

# RGB values for our scale
black = np.array([0, 0, 0])
beaver_orange = np.array([220, 68, 5]) / 255

# we only loop through three values -- the fourth is alpha, which can stay as 1
for i in range(3):
    # RGB values scaled from black to beaver orange
    vals[:, i] = np.linspace(black[0], beaver_orange[i], N)

# we can use beavs as a custom colormap!
beavs = ListedColormap(vals)


print()
print("Maximum ρ for SMM: %.3f, for OCI: %.3f"%(np.max(spec_rad_acc), np.max(spec_rad_oci)))
print()

c, mfp = np.meshgrid(c_range, mfp_range)


# cmaps =  ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#                       'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#                       'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
#                       'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
#                       'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
#                       'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
#                       'berlin', 'managua', 'vanimo', 'twilight', 'twilight_shifted', 'hsv',
#                       'tab20c',
#                       'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
#                       'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
#                       'tab20c']
# for i in range(len(cmaps)):
# name = cmaps[i]

name = 'inferno'
axi_font_size = 13

fig, (axs) = plt.subplots(ncols=3, layout="constrained")
ax1 = axs[1]
ax2 = axs[0]
ax3 = axs[2]
surf = ax1.contourf(mfp, c, spec_rad_acc, levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)

surf2 = ax2.contourf(mfp, c, spec_rad_oci, levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)

cb1 = fig.colorbar(surf2, ax=axs[1], ticks=[0, .2, .4, .6, .8, 1.0])
cb1.set_label(label=r'$\rho_e$',size=axi_font_size)

surf.set_edgecolor("face")
surf2.set_edgecolor("face")

ax1.set_xscale('log')
ax2.set_xscale('log')

ax1.set_xlabel(r"$\delta$", fontsize=axi_font_size)
ax2.set_ylabel(r"$c$", fontsize=axi_font_size)
ax2.set_xlabel(r"$\delta$", fontsize=axi_font_size)

ax1.set_title(r"2$^{nd}$ Moment Method")
ax2.set_title("OCI")
#ax3.set_title(r"$\rho$ Improvement")

rho_acc = spec_rad_oci/spec_rad_acc # computing the ratio

print("max ratio speedup: %.3f"%np.max(rho_acc))

rho_acc = np.ma.masked_where(rho_acc <= 0, rho_acc) # masking 0s

surf_acc = ax3.contourf(mfp, c, rho_acc, locator=ticker.LogLocator(subs='all'),  levels=100, cmap=mpl.colormaps[name], antialiased=True)
speed = ax3.contour(mfp, c, rho_acc, levels=[1.0], colors=('w',), linewidths=(3,))
ax3.clabel(speed, fmt='%2.1f', colors='w', fontsize=20)
surf_acc.set_edgecolor("face")
ax3.set_xlabel(r"$\delta$", fontsize=axi_font_size)
ax3.set_xscale('log')
cb2 = fig.colorbar(surf_acc, ax=axs.ravel().tolist(), ticks=[10e-1, 10e0, 10e1, 10e2, 10e3])
cb2.set_label(label=r'$\rho_{OCI}/\rho_{SMM}$',size=axi_font_size, weight='bold')


plt.gcf().set_size_inches(8, 4)
ax2.label_outer()
ax3.label_outer()
plt.savefig("smm_acc.pdf")
#plt.show()



exit()

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