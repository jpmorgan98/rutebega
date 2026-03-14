import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def main(npz_file="numerical_exp_2d_compare.npz"):
    data = np.load(npz_file)

    mfp_range   = data["mfp"]
    c_range     = data["c"]
    rho_oci     = data["rho_oci"]
    rho_p1      = data["rho_p1"]
    rho_upwind  = data["rho_upwind"]

    it_oci      = data.get("it_oci", None)
    it_p1       = data.get("it_p1", None)
    it_upwind   = data.get("it_upwind", None)

    print()
    print("Loaded:", npz_file)
    print("Max ρ: OCI = %.3f, P1 = %.3f, UP = %.3f" %
          (np.nanmax(rho_oci), np.nanmax(rho_p1), np.nanmax(rho_upwind)))
    if it_oci is not None:
        print("Max it: OCI = %d, P1 = %d, UP = %d" %
              (int(np.nanmax(it_oci)), int(np.nanmax(it_p1)), int(np.nanmax(it_upwind))))
    print()

    # -------------------------
    # plots: 3 panels
    # -------------------------
    C, D = np.meshgrid(c_range, mfp_range)

    fig, axs = plt.subplots(ncols=3, figsize=(12, 4), layout="constrained")

    global_min = np.min((rho_oci, rho_p1, rho_upwind))

    s1 = axs[0].contourf(D, C, rho_oci, levels=100, vmin=global_min, vmax=1.0,
                         cmap=cm.viridis, antialiased=True)
    s2 = axs[1].contourf(D, C, rho_p1, levels=100, vmin=global_min, vmax=1.0,
                         cmap=cm.viridis, antialiased=True)
    s3 = axs[2].contourf(D, C, rho_upwind, levels=100, vmin=global_min, vmax=1.0,
                         cmap=cm.viridis, antialiased=True)
    s1.set_edgecolor("face")
    s2.set_edgecolor("face")
    s3.set_edgecolor("face")

    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\delta$")
    axs[0].set_ylabel(r"$c$")

    axs[0].set_title("Normal OCI")
    axs[1].set_title("OCI + 2nd moment + P1 closure")
    axs[2].set_title("OCI + 2nd moment + upwind closure")

    fig.colorbar(s3, ax=axs.ravel().tolist(), label=r'$\rho_e$')
    plt.savefig("rho_compare_oci_p1_upwind.pdf")
    plt.show()

    # -------------------------
    # ratio plots vs OCI
    # -------------------------
    ratio_p1 = rho_oci / np.maximum(rho_p1, 1e-14)
    ratio_up = rho_oci / np.maximum(rho_upwind, 1e-14)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")

    r1 = axs[0].contourf(D, C, ratio_p1, levels=100, cmap=cm.viridis, antialiased=True)
    r2 = axs[1].contourf(D, C, ratio_up, levels=100, cmap=cm.viridis, antialiased=True)

    r1.set_edgecolor("face")
    r2.set_edgecolor("face")

    axs[0].contour(D, C, ratio_p1, levels=[1.0], colors=('w',), linewidths=(3,))
    axs[1].contour(D, C, ratio_up, levels=[1.0], colors=('w',), linewidths=(3,))

    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\delta$")
    axs[0].set_ylabel(r"$c$")

    axs[0].set_title(r"$\rho_{OCI}/\rho_{P1}$")
    axs[1].set_title(r"$\rho_{OCI}/\rho_{UP}$")

    fig.colorbar(r2, ax=axs.ravel().tolist())
    plt.savefig("rho_ratio_oci_over_p1_and_upwind.pdf")
    plt.show()


if __name__ == "__main__":
    main("numerical_exp_2d_compare.npz")