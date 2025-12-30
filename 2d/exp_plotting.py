import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------- load ----------
npz_name = "twoD_exp.npz"
with np.load(npz_name) as data:
    delta = data["mfp"]                  # your script saved mfp=delta
    c = data["c"]
    rho_oci = data["spec_rad_oci"]
    rho_yavuz = data["spec_rad_yavuz"]
    rho_rescale = data["spec_rad_rescale"]

# ---------- shape sanity ----------
# Your run script builds arrays as (len(c), len(delta)).
# If anything got transposed in a different run, fix it here.
expected = (c.size, delta.size)
if rho_oci.shape != expected and rho_oci.shape == (delta.size, c.size):
    rho_oci = rho_oci.T
    rho_yavuz = rho_yavuz.T
    rho_rescale = rho_rescale.T

if rho_oci.shape != expected:
    raise ValueError(
        f"Unexpected array shapes. Expected {expected} (c, delta). "
        f"Got rho_oci.shape={rho_oci.shape}, c.size={c.size}, delta.size={delta.size}"
    )

rho_yavuz = rho_yavuz.astype(float)          # ensure it's float
rho_yavuz[rho_yavuz > 1.0] = np.nan          # mark "None" locations as NaN
cmap = cm.viridis.copy()
cmap.set_bad(color="red")

# meshgrid that matches Z = (c, delta)
DELTA, C = np.meshgrid(delta, c, indexing="xy")   # shapes (c, delta)

# ---------- plot ----------
fig, axs = plt.subplots(
    ncols=3, figsize=(10.5, 3.6),
    constrained_layout=True, sharex=True, sharey=True
)

fields = [
    (rho_oci, "OCI"),
    (rho_yavuz, "SMM (yavuz)"),
    (rho_rescale, "SMM (rescale)"),
]

levels = 100
vmin, vmax = 0.0, 1.0
last_cf = None

for ax, (Z, title) in zip(axs, fields):
    Zm = np.ma.masked_invalid(Z) 
    cf = ax.pcolormesh(DELTA, C, Zm, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
    last_cf = cf
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(r"$\delta$")

axs[0].set_ylabel(r"$c$")

# one shared colorbar
cbar = fig.colorbar(last_cf, ax=axs, pad=0.02)
cbar.set_label(r"$\rho_e$")

plt.savefig("spec_rad_3panel.pdf", dpi=300)
plt.show()

# Pick which error metric to plot: "l2" or "linf"
metric = "l2"

data = np.load("twoD_exp.npz")
# Expected arrays saved from your run:
# err_phi_rescale_l2, err_psi_rescale_l2, err_phi_yavuz_l2, err_psi_yavuz_l2
err_phi_rescale = data[f"err_phi_rescale_{metric}"]  # (Nc, Nd)
err_psi_rescale = data[f"err_psi_rescale_{metric}"]  # (Nc, Nd)
err_phi_yavuz   = data[f"err_phi_yavuz_{metric}"]    # (Nc, Nd)
err_psi_yavuz   = data[f"err_psi_yavuz_{metric}"]    # (Nc, Nd)

from matplotlib.colors import LogNorm

# ---- plotting helper ----
def plot_pair(title, err_phi, err_psi):

    print()
    print(err_phi)
    print()
    # Put delta on x (log), c on y
    X, Y = np.meshgrid(delta, c)  # X,Y shape (Nc, Nd)

    # Avoid log10 issues if any tiny zeros slipped in
    #eps = 1e-300
    #Zphi = np.log10(np.maximum(err_phi, eps))
    #Zpsi = np.log10(np.maximum(err_psi, eps))

    tiny = 1e-300  # avoid zeros
    E = np.maximum(err_phi, tiny)  # or any error array
    norm_phi = LogNorm(vmin=E.min(), vmax=E.max())
    E = np.maximum(err_psi, tiny)  # or any error array
    norm_psi = LogNorm(vmin=E.min(), vmax=E.max())

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    im0 = axs[0].pcolormesh(X, Y, err_phi, shading="auto", norm=norm_phi)
    axs[0].set_xscale("log")
    axs[0].set_title(r"$\log_{10}$ error in $\phi$")
    axs[0].set_xlabel(r"$\delta$")
    axs[0].set_ylabel(r"$c$")
    cb0 = fig.colorbar(im0, ax=axs[0])
    cb0.set_label(r"$\log_{10}(\mathrm{rel\ error})$")

    im1 = axs[1].pcolormesh(X, Y, err_psi, shading="auto", norm=norm_psi)
    axs[1].set_xscale("log")
    axs[1].set_title(r"$\log_{10}$ error in $\psi$")
    axs[1].set_xlabel(r"$\delta$")
    axs[1].set_ylabel(r"$c$")
    cb1 = fig.colorbar(im1, ax=axs[1])
    cb1.set_label(r"$\log_{10}(\mathrm{rel\ error})$")

    fig.suptitle(f"{title} (metric: {metric})")
    return fig, axs

# ---- figure 1: rescale ----
plot_pair("Rescale vs transport", err_phi_rescale, err_psi_rescale)
plt.savefig(f"err_rescale_{metric}.png", dpi=200)

# ---- figure 2: yavuz ----
plot_pair("Yavuz vs transport", err_phi_yavuz, err_psi_yavuz)
plt.savefig(f"err_yavuz_{metric}.png", dpi=200)

plt.show()
