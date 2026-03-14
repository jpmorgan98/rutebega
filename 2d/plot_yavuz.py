#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmcrameri.cm as cmc
import matplotlib.colors as colors

cmap = cm.inferno #cmc.batlow

import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 16
#plt.rcParams["legend.fontsize"] = 15
plt.rcParams["figure.titlesize"]= 16

def _load_item_if_object_array(x):
    return x.item() if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1 else x


# Colorblind-safe palette (Okabe–Ito) + distinctive linestyles
STYLE = {
    # Second-moment
    "phi_second_simple":      dict(color="#0072B2", linestyle="--", linewidth=2.0),  # blue
    "phi_second_additive":    dict(color="#009E73", linestyle="-.", linewidth=2.0),  # bluish green

    # Diffusion
    "phi_diffusion_simple":   dict(color="#D55E00", linestyle=":",  linewidth=2.2),  # vermillion
    "phi_diffusion_additive": dict(color="#CC79A7", linestyle=(0, (3, 1, 1, 1)), linewidth=2.0),  # reddish purple (dash-dot-dot)

    # Reference solutions
    "phi_transport":          dict(color="#000000", linestyle="-",  linewidth=2.6),  # black solid
    "phi_lo":                 dict(color="#E69F00", linestyle=(0, (5, 2)), linewidth=2.2),  # orange dashed

    "phi_second_additive2":   dict(color="#56B4E9", linestyle=":", linewidth=2.0)
}



def midline_index(y, y_index=None):
    y = np.asarray(y).reshape(-1)
    if y_index is not None:
        j = int(y_index)
    else:
        j = int(np.argmin(np.abs(y - np.mean(y))))
    return max(0, min(y.size - 1, j))


def centers_to_edges(c):
    c = np.asarray(c).reshape(-1)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    dc = np.diff(c)
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * dc[0]
    edges[-1] = c[-1] + 0.5 * dc[-1]
    return edges


def mu_to_edges(mu_sorted):
    mu_sorted = np.asarray(mu_sorted).reshape(-1)
    edges = np.empty(mu_sorted.size + 1, dtype=float)
    if mu_sorted.size == 1:
        edges[0], edges[1] = -1.0, 1.0
        return edges
    edges[1:-1] = 0.5 * (mu_sorted[:-1] + mu_sorted[1:])
    edges[0] = -1.0
    edges[-1] = 1.0
    edges = np.maximum.accumulate(edges)  # ensure monotone
    return edges


def psi_midline_x_mu(psi, j_mid):
    """
    Returns psi_mid with shape (N_dir, Nx) = psi(x, y_mid, mu).

    Supports:
      psi: (Nx,Ny,N_dir,4) -> corner-average to (Nx,Ny,N_dir)
      psi: (Nx,Ny,N_dir)
    """
    psi = np.asarray(psi)
    if psi.ndim == 4:
        psi_avg = psi.mean(axis=3)          # (Nx,Ny,N_dir)
    elif psi.ndim == 3:
        psi_avg = psi
    else:
        raise ValueError(f"psi expected 3D or 4D, got shape {psi.shape}")

    # (Nx,Ny,N_dir) -> (N_dir,Nx) at y_mid
    return psi_avg[:, j_mid, :].T


def get_iter_map(data):
    """
    Map phi_*/psi_* keys -> iteration counts based on how your driver appends `its`.

    Your driver append order (from your first script):
      0: second_simple
      1: second_additive
      2: diffusion_simple
      3: diffusion_additive
      4: transport
    """
    it_map = {}

    if "its" not in data.files:
        return it_map

    its = np.asarray(data["its"]).reshape(-1)

    def it_at(i):
        if i < 0 or i >= its.size:
            return None
        try:
            return int(its[i])
        except Exception:
            return None

    # scalar
    it_map["phi_second_simple"]      = it_at(0)
    it_map["phi_second_additive"]    = it_at(1)
    it_map["phi_diffusion_simple"]   = it_at(2)
    it_map["phi_diffusion_additive"] = it_at(3)
    it_map["phi_transport"]          = it_at(4)
    it_map["phi_lo"]                 = None  # direct solve

    # angular
    it_map["psi_second_simple"]      = it_at(0)
    it_map["psi_second_additive"]    = it_at(1)
    it_map["psi_diffusion_simple"]   = it_at(2)
    it_map["psi_diffusion_additive"] = it_at(3)
    it_map["psi_transport"]          = it_at(4)

    return it_map


def label_with_it(label, it):
    return f"{label} ({it})" if it is not None else f"{label}"

def rel_L2_xmu(psi_mid, psiT_mid, w_sorted, dx):
    """
    Relative weighted L2 norm over (x,mu) on the midline:
      ||e|| / ||psiT|| where sums use quadrature weights and dx.
    psi_mid, psiT_mid: (N_dir, Nx) with same mu ordering as w_sorted
    w_sorted: (N_dir,)
    """
    e = psi_mid - psiT_mid
    num2 = np.sum((e * e) * w_sorted[:, None]) * dx
    den2 = np.sum((psiT_mid * psiT_mid) * w_sorted[:, None]) * dx
    return np.sqrt(num2 / (den2 + 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="yavuz_slab.npz", help="npz file (default: yavuz_slab.npz)")
    ap.add_argument("--y_index", type=int, default=None, help="explicit y-index for midline (default: closest to y-mid)")
    ap.add_argument("--show", action="store_true", help="call plt.show()")
    args = ap.parse_args()

    data = np.load(args.file, allow_pickle=True)

    if "mesh" not in data.files or "ang" not in data.files:
        raise ValueError("Expected 'mesh' and 'ang' in the npz.")

    mesh = _load_item_if_object_array(data["mesh"])
    ang  = _load_item_if_object_array(data["ang"])

    x  = np.asarray(mesh["x"])
    y  = np.asarray(mesh["y"])
    mu = np.asarray(ang["mu"])

    j_mid = midline_index(y, args.y_index)
    y_mid_val = y[j_mid]

    it_map = get_iter_map(data)

    # ----------------------------
    # FIGURE A: scalar flux (two panels)
    # ----------------------------
    diffusion_methods = [
        ("phi_lo",                 "diffusion sol"),
        ("phi_transport",          "transport"),
        ("phi_diffusion_simple",   "upwind"),
        ("phi_diffusion_additive", r"P$_1$"),
    ]
    second_moment_methods = [
        ("phi_lo",                 "diffusion sol"),
        ("phi_transport",          "transport"),
        ("phi_second_simple",      "upwind"),
        ("phi_second_additive",    r"P$_1$"),
    ]

    figA, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    panels = [
        (axes[0], "Diffusion methods", diffusion_methods),
        (axes[1], "Second-moment methods", second_moment_methods),
    ]

    for ax, title, methods in panels:
        for key, pretty in methods:
            if key not in data.files:
                continue
            phi = np.asarray(data[key])
            if phi.ndim != 2:
                continue
            it = it_map.get(key, None)
            style = STYLE.get(key, {})
            ax.plot(x, phi[:, j_mid], label=label_with_it(pretty, it), **style)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel(r"$\phi(x, y_{mid})$")
    #figA.suptitle(f"Midline scalar flux (y ≈ {y_mid_val:.6g}, index {j_mid})", y=1.02)
    figA.tight_layout()

    # ----------------------------
    # FIGURE B: 2D angular flux subplots (x vs mu colormap)
    #   Top row: shared colorbar
    #   Bottom row: shared colorbar
    # ----------------------------
    psi_keys = [
        ("psi_second_simple",      "second moment simple"),
        ("psi_second_additive",    "second moment additive"),
        ("psi_diffusion_simple",   "diffusion simple"),
        ("psi_diffusion_additive", "diffusion additive"),
        ("psi_transport",          "transport"),
    ]
    psi_keys = [(k, lbl) for (k, lbl) in psi_keys if k in data.files]

    if not psi_keys:
        print("No psi_* arrays found in the npz; skipping angular flux figure.")
    else:
        psi_err_keys = [
            ("psi_second_simple",      "SMM upwind"),
            ("psi_second_additive",    r"SMM P$_1$"),
            ("psi_diffusion_simple",   "Diff Acc upwind"),
            ("psi_diffusion_additive", r"Diff Acc P$_1$"),
        ]
        psi_err_keys = [(k, lbl) for (k, lbl) in psi_err_keys if k in data.files]

        if not psi_err_keys:
            print("No psi_* methods (besides transport) found; skipping angular error figure.")
        else:
            mu_order  = np.argsort(mu)
            mu_sorted = mu[mu_order]

            w = np.asarray(ang["w"])
            w_sorted = w[mu_order]
            dx = float(mesh["dx"])

            # transport reference on the midline, sorted in mu
            psiT_mid = psi_midline_x_mu(data["psi_transport"], j_mid)[mu_order, :]  # (N_dir, Nx)

            # --- precompute all error fields + L2 ---
            err_fields = []
            l2_vals = []
            for key, pretty in psi_err_keys:
                psi_mid = psi_midline_x_mu(data[key], j_mid)[mu_order, :]  # (N_dir, Nx)
                err = np.abs(psi_mid - psiT_mid)  # (Nmu, Nx)
                err_fields.append(err)
                l2_vals.append(rel_L2_xmu(psi_mid, psiT_mid, w_sorted, dx))

            n = len(psi_err_keys)
            ncols = 2
            nrows = int(np.ceil(n / ncols))

            figB, axesB = plt.subplots(
                nrows, ncols,
                figsize=(12, 3.8 * nrows),
                squeeze=False,
                sharex=True, sharey=True
            )

            # --- build row-wise normalization (top vs bottom) ---
            nlevels = 100

            def row_norm_and_levels(row_idx):
                idxs = [i for i in range(n) if (i // ncols) == row_idx]
                if not idxs:
                    return colors.Normalize(vmin=0.0, vmax=1.0), np.linspace(0.0, 1.0, nlevels)
                vmin = 0.0
                vmax = max(float(np.nanmax(err_fields[i])) for i in idxs)
                if not np.isfinite(vmax) or vmax <= 0.0:
                    vmax = 1.0
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                levels = np.linspace(vmin, vmax, nlevels)
                return norm, levels

            norm_top, levels_top = row_norm_and_levels(0)
            norm_bot, levels_bot = row_norm_and_levels(1) if nrows > 1 else (norm_top, levels_top)

            mappable_top = None
            mappable_bot = None

            for idx, ((key, pretty), err, l2rel) in enumerate(zip(psi_err_keys, err_fields, l2_vals)):
                r = idx // ncols
                c = idx % ncols
                ax = axesB[r][c]

                # choose norm/levels by row
                if r == 0:
                    norm_use, levels_use = norm_top, levels_top
                else:
                    norm_use, levels_use = norm_bot, levels_bot

                cs = ax.contourf(
                    x, mu_sorted, err,
                    levels=levels_use,
                    cmap=cmap,
                    norm=norm_use,
                    antialiased=True
                )
                cs.set_edgecolor("face")

                if r == 0 and mappable_top is None:
                    mappable_top = cs
                if r != 0 and mappable_bot is None:
                    mappable_bot = cs

                ax.set_title(f"{pretty}", fontsize=10)
                ax.text(
                    0.02, 0.98, rf"$\|\epsilon\|_2 = {l2rel:.2e}$",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
                )

            # turn off unused axes
            for k in range(n, nrows * ncols):
                axesB[k // ncols][k % ncols].set_axis_off()

            # --- two separate shared colorbars: one for top row, one for bottom row ---
            # Only include "active" axes in each row for the colorbar
            top_axes = [axesB[0, j] for j in range(ncols) if (0 * ncols + j) < n and axesB[0, j].axison]
            if mappable_top is not None and top_axes:
                figB.colorbar(
                    mappable_top,
                    ax=top_axes,
                    shrink=0.92,
                    label=r"$\epsilon$ SMM)",
                )

            if nrows > 1:
                bot_axes = [axesB[1, j] for j in range(ncols) if (1 * ncols + j) < n and axesB[1, j].axison]
                if mappable_bot is not None and bot_axes:
                    figB.colorbar(
                        mappable_bot,
                        ax=bot_axes,
                        shrink=0.92,
                        label=r"$\epsilon$ Diff Acc",
                    )

            # axis labels
            for r in range(nrows):
                axesB[r, 0].set_ylabel(r"$\mu$")
            for c in range(ncols):
                axesB[nrows - 1, c].set_xlabel(r"$x$")

    figA.savefig("sf_midline_yavuz_s64.pdf")
    figB.savefig("af_midline_yavuz_s64.pdf")

    # ----------------------------
    # FIGURE C: scalar flux no transport scatter (two panels)
    # ----------------------------

    data_nt = np.load("yavuz_slab_noTransScater.npz", allow_pickle=True)

    diffusion_methods = [
        ("phi_lo",                 "diffusion sol"),
        ("phi_diffusion_simple",   "upwind"),
        ("phi_diffusion_additive", r"P$_1$"),
        #("phi_transport",          "transport"),
    ]
    second_moment_methods = [
        ("phi_lo",                 "diffusion sol"),
        ("phi_second_simple",      "upwind"),
        ("phi_second_additive",    r"P$_1$",),
        #("phi_transport",          "transport"),
    ]

    figC, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)

    panels = [
        (axes[0], "Diffusion methods", diffusion_methods),
        (axes[1], "Second-moment methods", second_moment_methods),
    ]

    for ax, title, methods in panels:
        for key, pretty in methods:
            if key not in data.files:
                continue
            phi = np.asarray(data_nt[key])
            if phi.ndim != 2:
                continue

            it = it_map.get(key, None)
            style = STYLE.get(key, {})
            ax.plot(x, phi[:, j_mid], label=label_with_it(pretty, it), **style)

        trans_key = "phi_transport"
        pretty = "transport"
        phi = np.asarray(data[trans_key])
        it = it_map.get(trans_key, None)
        style = STYLE.get(trans_key, {})
        ax.plot(x, phi[:, j_mid], label=label_with_it(pretty, it), **style)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel(r"$\phi(x, y_{mid})$")
    #figA.suptitle(f"Midline scalar flux (y ≈ {y_mid_val:.6g}, index {j_mid})", y=1.02)

    ax = axes[2]
    key = "phi_second_additive"
    it = it_map.get(key, None)
    style = STYLE.get(key, {})
    ax.plot(x, np.asarray(data[key])[:, j_mid],    label= r"P$_1$ HO Scattering", **style)
    style = STYLE.get("phi_second_additive2", {})
    ax.plot(x, np.asarray(data_nt[key])[:, j_mid], label= r"P$_1$ No HO Scattering", **style)
    trans_key = "phi_transport"
    pretty = "transport"
    style = STYLE.get(trans_key, {})
    ax.plot(x, np.asarray(data[trans_key])[:, j_mid], label= r"Transport", **style)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlabel("x")
    ax.set_title("Scattering?")

    figC.tight_layout()
    figC.savefig("no_scatter_yavuz.pdf")


    if args.show:
        plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    main()

    
