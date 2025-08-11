import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

import matplotlib as mpl


#mfp=mfp_range, c=c_range, spec_rad_oci=spec_rad_oci, spec_rad_pacc=spec_rad_pacc
with np.load('td_exp.npz') as data:
    mfp_range = data['mfp']
    c_range = data['c']
    mft_range = data['mft']
    spec_rad_oci = data['spec_rad_oci']
    spec_rad_acc = data['spec_rad_pacc']

rho_acc = spec_rad_oci/spec_rad_acc # computing the ratio

print(spec_rad_oci[-1,-1,:])
print(spec_rad_acc[-1,-1,:])
print(rho_acc[-1,-1,:])


print(mfp_range)
print(c_range)
print(mft_range)

N_mfp = mfp_range.size
N_c = c_range.size
N_mft = mft_range.size

fig, (axs) = plt.subplots(nrows=3, layout="constrained", sharex=True)

mfp_plot_in = [0, 4, 7]
c_plot_in = [2,4,5]

for j in range(3):
    ax = axs[j]
    for i in range(3):
        ax.plot(mft_range, spec_rad_oci[mfp_plot_in[j], c_plot_in[i], :], 'r' ,label="t, c%.1f"%c_range[c_plot_in[i]])
        ax.plot(mft_range, spec_rad_acc[mfp_plot_in[j], c_plot_in[i], :], 'k' ,label="a, c%.1f"%c_range[c_plot_in[i]])
    ax.legend()
    ax.set_title("δ = %.2f"%mfp_range[mfp_plot_in[j]])
    ax.set_ylabel(r"$\rho$")
    #ax.grid()
    ax.set_xscale("log")
    ax.set_ylim((0,1.25))

axs[0].set_xlabel("τ")

plt.show()


exit()


print()
print("Maximum ρ for SMM: %.3f, for OCI: %.3f"%(np.max(spec_rad_acc), np.max(spec_rad_oci)))
print()

c, mfp = np.meshgrid(c_range, mfp_range)

name = 'inferno'


rho_acc = np.ma.masked_where(rho_acc <= 0, rho_acc) # masking 0s

for t in range(len(mft_range)):

    fig, (axs) = plt.subplots(ncols=3, layout="constrained")
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    surf = ax1.contourf(mfp, c, spec_rad_acc[:,:,t], levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)

    surf2 = ax2.contourf(mfp, c, spec_rad_oci[:,:,t], levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)

    fig.colorbar(surf2, ax=axs[1], label=r'$\rho_e$', ticks=[0, .2, .4, .6, .8, 1.0])

    surf.set_edgecolor("face")
    surf2.set_edgecolor("face")

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    ax1.set_xlabel(r"$\delta$")
    ax1.set_ylabel("MFT %.1e \n c"%mft_range[t])
    ax2.set_xlabel(r"$\delta$")

    ax1.set_title(r"2$^{nd}$ Moment Method")
    ax2.set_title("OCI")
    ax3.set_title(r"$\rho$ Improvement")

    print("max ratio speedup: %.3f"%np.max(rho_acc[:,:,t]))

    surf_acc = ax3.contourf(mfp, c, rho_acc[:,:,t], locator=ticker.LogLocator(subs='all'),  levels=100, cmap=mpl.colormaps[name], antialiased=True)
    speed = ax3.contour(mfp, c, rho_acc[:,:,t], levels=[1.0], colors=('w',), linewidths=(3,))
    ax3.clabel(speed, fmt='%2.1f', colors='w', fontsize=20)
    surf_acc.set_edgecolor("face")
    ax3.set_xlabel(r"$\delta$")
    ax3.set_xscale('log')
    fig.colorbar(surf_acc, ax=axs.ravel().tolist(), label=r'$r$', ticks=[10e-1, 10e0, 10e1, 10e2, 10e3])


    plt.gcf().set_size_inches(8, 4)
    ax2.label_outer()
    ax3.label_outer()
    plt.savefig("smm_acc%f.pdf"%mft_range[t])
    #plt.show()


c, mft = np.meshgrid(c_range, mft_range)
for i in range(len(mfp_range)):

    fig, (axs) = plt.subplots(ncols=3, layout="constrained")
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    surf = ax1.contourf(mft, c, np.transpose(spec_rad_acc[i,:,:]), levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)

    surf2 = ax2.contourf(mft, c, spec_rad_oci[i,:,:].T, levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)

    fig.colorbar(surf2, ax=axs[1], label=r'$\rho_e$')

    surf.set_edgecolor("face")
    surf2.set_edgecolor("face")

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel("δ %.1e \n c"%mfp_range[i])
    ax2.set_xlabel(r"$\tau$")

    ax1.set_title(r"2$^{nd}$ Moment Method")
    ax2.set_title("OCI")
    ax3.set_title(r"$\rho$ Improvement")

    print("max ratio speedup: %.3f"%np.max(rho_acc[i,:,:]))

    # space, scattering ratio, time

    surf_acc = ax3.contourf(mft, c, rho_acc[i,:,:].T, locator=ticker.LogLocator(subs='all'),  levels=100, cmap=mpl.colormaps[name], antialiased=True)
    speed = ax3.contour(mft, c, rho_acc[i,:,:].T, levels=[1.0], colors=('w',), linewidths=(3,))
    ax3.clabel(speed, fmt='%2.1f', colors='w', fontsize=20)
    surf_acc.set_edgecolor("face")
    ax3.set_xlabel(r"$\tau$")
    ax3.set_xscale('log')
    fig.colorbar(surf_acc, ax=axs.ravel().tolist(), label=r'$r$')


    plt.gcf().set_size_inches(8, 4)
    ax2.label_outer()
    ax3.label_outer()
    plt.savefig("td_results/smm_acc%f.pdf"%mfp_range[i])
    #plt.show()