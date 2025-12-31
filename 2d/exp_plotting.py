import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cmcrameri.cm as cmc
from matplotlib.patches import Rectangle


def add_bad_note(fig, cbar, cmap, note=r"bad = masked/invalid",
                               ref_axes=None, note_frac=0.1, fontsize=10):
    """
    Shrink the colorbar so there is room for a note *inside* the plot height.
    The note block's bottom is aligned with the bottom of the reference axes
    (i.e. the plot x-axis spine position in figure coords).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    cbar : matplotlib.colorbar.Colorbar
    cmap : matplotlib.colors.Colormap
    note : str
    ref_axes : Axes or list[Axes] or None
        Axes to align to. If None, uses cbar.mappable.axes.
    note_frac : float
        Note height as a fraction of the reference axes height.
    """
    # finalize constrained_layout positions before we move anything
    fig.canvas.draw()

    # get "bad" RGBA
    try:
        bad = cmap.get_bad()
    except Exception:
        bad = getattr(cmap, "_rgba_bad", (1.0, 0.0, 1.0, 1.0))

    # choose reference axes and y0 alignment
    if ref_axes is None:
        ref_axes = cbar.mappable.axes
    if isinstance(ref_axes, (list, tuple)):
        y0_ref = min(ax.get_position().y0 for ax in ref_axes)
        h_ref  = min(ax.get_position().height for ax in ref_axes)
    else:
        pos_ref = ref_axes.get_position()
        y0_ref, h_ref = pos_ref.y0, pos_ref.height

    # current colorbar axis position (figure coords)
    cax = cbar.ax
    pos = cax.get_position()

    # reserve a note band *inside* the plot height
    note_h = note_frac * h_ref
    new_y0 = y0_ref + note_h
    new_h  = pos.y1 - new_y0
    if new_h <= 0:
        raise ValueError("Note band too tall for this colorbar; reduce note_frac.")

    # shrink colorbar to make room below
    cax.set_position([pos.x0, new_y0, pos.width, new_h])

    # create a tiny axes for the note exactly between x-axis and colorbar
    nax = fig.add_axes([pos.x0, y0_ref, pos.width, note_h])
    nax.set_axis_off()

    # swatch + text in note-axes coords
    nax.add_patch(Rectangle((0.08, 0.05), 1.0, 1.0,
                            transform=nax.transAxes, clip_on=False,
                            facecolor=bad, edgecolor="k", linewidth=0.5))
    nax.text(2.5, 0.05, note,
             transform=nax.transAxes, clip_on=False, rotation=90,
             ha="left", va="bottom", fontsize=fontsize)





# ---------- load ----------
npz_name = "twoD_exp2.npz"
with np.load(npz_name) as data:
    delta = data["mfp"]                  # your script saved mfp=delta
    c = data["c"]
    rho_oci = data["spec_rad_oci"]
    rho_yavuz = data["spec_rad_yavuz"]
    rho_rescale = data["spec_rad_rescale"]


# Pick which error metric to plot: "l2" or "linf"
metric = "linf"

data = np.load("twoD_exp2.npz")
# Expected arrays saved from your run:
# err_phi_rescale_l2, err_psi_rescale_l2, err_phi_yavuz_l2, err_psi_yavuz_l2
err_phi_rescale = data[f"err_phi_rescale_{metric}"]  # (Nc, Nd)
err_psi_rescale = data[f"err_psi_rescale_{metric}"]  # (Nc, Nd)
err_phi_yavuz   = data[f"err_phi_yavuz_{metric}"]    # (Nc, Nd)
err_psi_yavuz   = data[f"err_psi_yavuz_{metric}"]    # (Nc, Nd)

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

for i in range(rho_yavuz.shape[0]):
    for j in range(rho_yavuz.shape[1]):
        if float(rho_yavuz[i,j]) > 1.0:
            rho_yavuz[i,j] = np.nan          # mark "None" locations as NaN
            err_phi_yavuz[i,j] = np.nan
            err_psi_yavuz[i,j] = np.nan



cmap = cm.viridis.copy()
cmap.set_bad(color="orange")

cmap_b = cmc.batlow.copy()
cmap_b.set_bad(color="green")

# meshgrid that matches Z = (c, delta)
DELTA, C = np.meshgrid(delta, c, indexing="xy")   # shapes (c, delta)

# ---------- plot ----------
fig, axs = plt.subplots(
    ncols=3, figsize=(10.5, 3.6),
    constrained_layout=False, sharex=True, sharey=True
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

add_bad_note(fig, cbar, cmap, note=r"$\rho$>1.0")

plt.savefig("spec_rad_3panel.pdf", dpi=300)
plt.show()

from matplotlib.colors import LogNorm

# ---- plotting helper ----
def plot_pair(title, err_phi, err_psi):

    # Put delta on x (log), c on y
    X, Y = np.meshgrid(delta, c)  # X,Y shape (Nc, Nd)

    # Avoid log10 issues if any tiny zeros slipped in
    #eps = 1e-300
    #Zphi = np.log10(np.maximum(err_phi, eps))
    #Zpsi = np.log10(np.maximum(err_psi, eps))

    # tiny = 1e-300  # avoid zeros
    # E = np.maximum(err_phi, tiny)  # or any error array
    # norm_phi = LogNorm(vmin=E.min(), vmax=E.max())
    # E = np.maximum(err_psi, tiny)  # or any error array
    # norm_psi = LogNorm(vmin=E.min(), vmax=E.max())

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=False)

    im0 = axs[0].pcolormesh(X, Y, err_phi, shading="auto", cmap=cmap_b )#norm=norm_phi)
    axs[0].set_xscale("log")
    axs[0].set_title(r"$L_{\infty}\left(\phi\right)$")
    axs[0].set_xlabel(r"$\delta$")
    axs[0].set_ylabel(r"$c$")
    cb0 = fig.colorbar(im0, ax=axs[0])
    cb0.set_label(r"$\epsilon$")
    add_bad_note(fig, cb0, cmap_b, ref_axes=axs[0], note=r"$\rho$>1.0")

    im1 = axs[1].pcolormesh(X, Y, err_psi, shading="auto", cmap=cmap_b )#norm=norm_psi)
    axs[1].set_xscale("log")
    axs[1].set_title(r"$L_{\infty}\left(\psi\right)$")
    axs[1].set_xlabel(r"$\delta$")
    axs[1].set_ylabel(r"$c$")
    cb1 = fig.colorbar(im1, ax=axs[1])
    cb1.set_label(r"$\epsilon$")
    add_bad_note(fig, cb1, cmap_b, ref_axes=axs[1], note=r"$\rho$>1.0")

    #fig.suptitle(f"{title} (metric: {metric})")
    return fig, axs

# ---- figure 1: rescale ----
plot_pair("Rescale vs transport", err_phi_rescale, err_psi_rescale)
plt.savefig(f"err_rescale_{metric}.png", dpi=200)

# ---- figure 2: yavuz ----
plot_pair("Yavuz vs transport", err_phi_yavuz, err_psi_yavuz)
plt.savefig(f"err_yavuz_{metric}.png", dpi=200)

plt.show()
