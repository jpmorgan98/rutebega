import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from concurrent.futures import ProcessPoolExecutor, as_completed

from smom_2d import transport_2d_oci_spectral


def run_case(args):
    """
    Runs one (delta, c) point and returns indices + results.
    Keep all three methods inside the same process to reuse psi0.
    """
    (k, h, delta, c, base_seed, q, sigma, Lx, Ly, N_dir) = args

    omega_total = 2.0 * np.pi

    # avoid exactly c=1
    c_eff = min(float(c), 1.0 - 1e-12)

    sigma_s = sigma * c_eff
    sigma_a = sigma - sigma_s

    dx_target = delta / sigma
    Nx = max(6, int(round(Lx / dx_target)))
    Ny = Nx

    psi_iso = q / (sigma_a * omega_total)
    bc = dict(left=psi_iso, right=psi_iso, bottom=psi_iso, top=psi_iso)

    rng = np.random.default_rng(base_seed + 1000*k + h)
    psi0 = psi_iso * (1.0 + 0.2*(rng.random((Nx, Ny, N_dir, 4)) - 0.5))
    psi0 = np.maximum(psi0, 0.0)

    # (1) Normal OCI
    r1, i1, _ = transport_2d_oci_spectral(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, N_dir=N_dir,
        sig_t_val=sigma, sig_s_val=sigma_s, q_val=q,
        bc=bc,
        smm_acc=False,
        tol=1e-10, max_it=100,
        psi0=psi0,
        printer=False
    )

    # (2) OCI + second_moment + P1 closure
    r2, i2, _ = transport_2d_oci_spectral(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, N_dir=N_dir,
        sig_t_val=sigma, sig_s_val=sigma_s, q_val=q,
        bc=bc,
        smm_acc=True,
        update="yavuz",
        diff="second_moment",
        closure="additive",
        sor_val=0.2,
        tol=1e-10, max_it=100,
        psi0=psi0,
        printer=False
    )

    # (3) OCI + second_moment + upwind closure
    r3, i3, _ = transport_2d_oci_spectral(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, N_dir=N_dir,
        sig_t_val=sigma, sig_s_val=sigma_s, q_val=q,
        bc=bc,
        smm_acc=True,
        update="yavuz",
        diff="second_moment",
        closure="simple_upwind",
        sor_val=0.2,
        tol=1e-10, max_it=100,
        psi0=psi0,
        printer=False
    )

    def clip_r(r):
        return (min(r, 1.0) if np.isfinite(r) else np.nan)

    return (k, h, delta, c_eff, clip_r(r1), i1, clip_r(r2), i2, clip_r(r3), i3)


if __name__ == "__main__":
    # problem knobs
    q = 2.0
    sigma = 5.0
    Len = 10.0
    Lx = Len
    Ly = Len
    N_dir = 10

    N_mfp = 10
    N_c   = 15
    mfp_range = np.logspace(-1, 1, N_mfp)
    c_range   = np.linspace(0.3, 1, N_c)

    rho_oci    = np.zeros((N_mfp, N_c))
    rho_p1     = np.zeros((N_mfp, N_c))
    rho_upwind = np.zeros((N_mfp, N_c))

    it_oci    = np.zeros((N_mfp, N_c), dtype=int)
    it_p1     = np.zeros((N_mfp, N_c), dtype=int)
    it_upwind = np.zeros((N_mfp, N_c), dtype=int)

    base_seed = 1234

    # Build task list
    tasks = []
    for k, delta in enumerate(mfp_range):
        for h, c in enumerate(c_range):
            tasks.append((k, h, float(delta), float(c), base_seed, q, sigma, Lx, Ly, N_dir))

    # Choose workers (often = number of physical cores; adjust as needed)
    max_workers = min(os.cpu_count() or 1, len(tasks))

    print(f"Running {len(tasks)} cases with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_case, t) for t in tasks]
        for fut in as_completed(futures):
            k, h, delta, c_eff, r1, i1, r2, i2, r3, i3 = fut.result()

            rho_oci[k, h]    = r1
            rho_p1[k, h]     = r2
            rho_upwind[k, h] = r3
            it_oci[k, h]     = i1
            it_p1[k, h]      = i2
            it_upwind[k, h]  = i3

            print(f"δ={delta:.3e}, c={c_eff:.3f}, "
                  f"OCI  ρ={r1:.6f} (it={i1:4d}), "
                  f"P1   ρ={r2:.6f} (it={i2:4d}), "
                  f"UP   ρ={r3:.6f} (it={i3:4d})")

    np.savez(
        "numerical_exp_2d_compare",
        mfp=mfp_range, c=c_range,
        rho_oci=rho_oci, rho_p1=rho_p1, rho_upwind=rho_upwind,
        it_oci=it_oci, it_p1=it_p1, it_upwind=it_upwind
    )

    print("\nMax ρ: OCI = %.3f, P1 = %.3f, UP = %.3f\n" %
          (np.nanmax(rho_oci), np.nanmax(rho_p1), np.nanmax(rho_upwind)))

    # plots (unchanged)
    C, D = np.meshgrid(c_range, mfp_range)
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4), layout="constrained")
    s1 = axs[0].contourf(D, C, rho_oci, levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)
    s2 = axs[1].contourf(D, C, rho_p1,  levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)
    s3 = axs[2].contourf(D, C, rho_upwind, levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)
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