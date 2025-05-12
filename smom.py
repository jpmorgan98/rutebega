import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import numba as nb

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)

# [1] M. Yavuz and E. Larsen (1989) *Spatial domain decomposition for Neutron transport problems* Transport Theory and Statistical Physics 18:2, 205-219
# [2] T. Palmer Private corispondence


#
# Global Variables
#

N_angle = 32
dxv = 1.0

Len = 25
N_cell = int(Len / dxv)

dx = dxv * np.ones(N_cell)

x_edge = np.linspace(0, Len, 2 * N_cell + 1)  # edges between half cells
x_mid = 0.5 * (x_edge[:-1] + x_edge[1:])  # middle of half cells

N = 2 * N_cell * N_angle

sigma = 1.5 * np.zeros(N_cell)
sigma_s = 0 * sigma
sigma_a = sigma - sigma_s

D = 1 / (3 * sigma)

tol = 1e-5
max_it = int(4000)

SET_acc = True
SET_acc_debug = False
printer = True
af_printer = False

[angles, weights] = np.polynomial.legendre.leggauss(N_angle)
mu_mid = 0.5 * (angles[:-1] + angles[1:])

source = 20 * np.ones(N)

af_l = 0
af_r = 0

sf_l = 0
sf_r = 0

J_l = 0
J_r = 0

second_l = 0
second_r = 0

Rem_l = 0
Rem_r = 0

bcl = "in"
bcr = "in"


#
# Method of Manufactured Solution
#


#
# Transport!
#


# @nb.jit
def Sbuild(i):
    S = np.zeros((2 * N_angle, 2 * N_angle))

    beta = sigma_s[i] * dx[i] / 4

    for p in range(N_angle):
        for j in range(N_angle):
            S[p * 2, j * 2] = beta * weights[j]
            S[p * 2 + 1, j * 2 + 1] = beta * weights[j]

    return S


# @nb.jit
def Ablock(mu, i):
    return np.array(
        (
            (np.abs(mu) / 2 + sigma[i] * dx[i] / 2, mu / 2),
            (-mu / 2, np.abs(mu) / 2 + sigma[i] * dx[i] / 2),
        )
    )


# @nb.jit
def buildA(i):
    A = np.zeros((2 * N_angle, 2 * N_angle))
    for a in range(N_angle):
        A[a * 2 : (a + 1) * 2, a * 2 : (a + 1) * 2] = Ablock(angles[a], i)

    S = Sbuild(i)

    A.shape == S.shape

    A = A - S

    return A


# @nb.jit
def b_neg(psi_rightBound, mu, i):
    b_n = np.array(
        (dx[i] / 4 * source[i], dx[i] / 4 * source[i] - mu * psi_rightBound)
    )
    return b_n


# @nb.jit
def b_pos(psi_leftBound, mu, i):
    b_p = np.array(
        (dx[i] / 4 * source[i] + mu * psi_leftBound, dx[i] / 4 * source[i])
    )
    return b_p


# @nb.jit
def buildb(aflux_edge, b, i):

    i = int(i)

    for m in range(N_angle):

        af_lb = aflux_edge[i * N_angle + m]
        if angles[m] > 0:
            af_lb = aflux_edge[i * N_angle + m]

            b[m * 2 : (m + 1) * 2] = b_pos(af_lb, angles[m], i)

        elif angles[m] < 0:
            af_rb = aflux_edge[(i + 1) * N_angle + m]

            b[m * 2 : (m + 1) * 2] = b_neg(af_rb, angles[m], i)


def compute_aflux_edge(aflux):
    # explicit implementation of upstream closure

    N_edges = int(N_cell + 1)
    af_edge = np.zeros(N_edges * N_angle)

    for i in range(N_edges):
        if i == 0:
            for m in range(N_angle):
                if bcl == "in":
                    af_edge[i * N_angle + m] = af_l
                elif bcl == "ref":
                    ref_angle_ord = int(N_angle - m - 1)
                    af_index = (0) * N_angle * 2 + ref_angle_ord * 2 + 1
                    af_edge[(i) * N_angle + m] = aflux[af_index]
                else:
                    print("Check bcl")

        elif i == N_cell:
            for m in range(N_angle):
                if bcr == "in":
                    af_edge[(i) * N_angle + m] = af_r
                elif bcr == "ref":
                    ref_angle_ord = int(N_angle - m - 1)
                    af_index = (i - 1) * N_angle * 2 + ref_angle_ord * 2 + 1
                    af_edge[(i) * N_angle + m] = aflux[af_index]
                else:
                    print("Check bcr")

        else:
            for m in range(N_angle):

                if angles[m] > 0:
                    af_index = (i - 1) * N_angle * 2 + m * 2 + 1
                    af_edge[(i) * N_angle + m] = aflux[af_index]

                elif angles[m] < 0:

                    af_index = (i) * N_angle * 2 + m * 2
                    af_edge[(i) * N_angle + m] = aflux[af_index]

    return af_edge


#
# Second moment equations from [1] and [2]
#


# @nb.jit
def compute_volume_moments(af):

    N_mom = 2 * N_cell

    zeroth = np.zeros(N_mom)
    first = np.zeros(N_mom)
    second = np.zeros((N_mom))

    for j in range(N_cell):
        for m in range(N_angle):
            af_index = j * N_angle * 2 + m * 2

            # left
            zeroth[2 * j] += weights[m] * af[af_index]
            first[2 * j] += weights[m] * angles[m] * af[af_index]
            second[2 * j] += (
                weights[m] * (0.5 * (3 * (angles[m] ** 2) - 1)) * af[af_index]
            )

            # right
            zeroth[2 * j + 1] += weights[m] * af[af_index + 1]
            first[2 * j + 1] += weights[m] * angles[m] * af[af_index + 1]
            second[2 * j + 1] += (
                weights[m] * (0.5 * (3 * (angles[m] ** 2) - 1)) * af[af_index + 1]
            )

    return (zeroth, first, second)


# @nb.jit
def build_smm_mat():
    # building the diffusion A mat
    # this system is *very* sparse, doing this as dense is silly but ok for now

    delta = 1 / 2
    gamma = 1 / 4

    A = np.zeros((2 * N_cell, 2 * N_cell))

    for i in range(N_cell):

        A[2*i,2*i] = (1 - delta) * D[i] / dx[i] + gamma + sigma_a[i] * dx[i] / 2
        A[2*i,2*i+1] = -(1 - delta) * D[i] / dx[i]

        A[2*i+1,2*i] = -(1 - delta) * D[i] / dx[i]
        A[2*i+1,2*i+1] = ((1 - delta) * D[i] / dx[i] + gamma + sigma_a[i] * dx[i] / 2)

        if i == N_cell - 1:  # right bound (no right of cell info)
            A[2*i, 2*(i-1)] = -delta * D[i-1] / dx[i-1]
            A[2*i, 2*(i-1)+1] = delta * D[i-1] / dx[i-1] - gamma

            # overwriting the last row
            if bcr == "ref":
                A[2*i+1, 2*i] = -(delta) * D[i] / dx[i]
                A[2*i+1, 2*i+1] = (delta) * D[i] / dx[i] + sigma_a[i] * dx[i] / 2

        elif i == 0:  # left bound (no left of cell info)

            A[2*i+1, 2*(i+1)] = -gamma + delta * D[i+1] / dx[i+1]
            A[2*i+1, 2*(i+1) + 1] = -delta * D[i+1] / dx[i+1]

            # rewriting first equation for residual based bounds
            #A[2*i,2*i] = .5 + dx[i]*sigma_a[i]/2 + D[i]/dx[i]
            #A[2*i,2*i+1] =  -D[i]/dx[i]

            # over writing the first row
            if bcl == "ref":
                A[2*i, 2*i] = (delta) * D[i] / dx[i] + sigma_a[i] * dx[i] / 2
                A[2*i, 2*i+1] = -(delta) * D[i] / dx[i]

        else:  # interior cell
            # cell j-1 info
            A[2*i, 2*(i-1)] = -delta * D[i-1] / dx[i-1]
            A[2*i, 2*(i-1)+1] = delta * D[i-1] / dx[i-1] - gamma
            # cell j+1 info=
            A[2*i+1, 2*(i+1)] = -gamma + delta * D[i+1] / dx[i+1]
            A[2*i+1, 2*(i+1)+1] = -delta * D[i+1] / dx[i+1]

    return A


# @nb.jit
def build_smm_b(second):
    delta = 0.5

    b_vec = np.zeros(2 * N_cell)
    for i in range(N_cell):
        if i == 0:  # left hand bound, using bcs
            b_l = dx[i]/2*source[i] + (J_l) +  2*D[i]/dx[i]*(second[2*i+1] - second[2*i])
            b_r = dx[i]/2*source[i] - (1-delta)*4*D[i]/dx[i]*(second[2*i+1] - second[2*i]) + (delta*2*D[i+1]) / dx[i+1]*(second[2*(i+1)+1]-second[2*(i+1)])

        elif i == N_cell - 1:  # right hand bcs
            b_l = dx[i]/2*source[i]+ (1-delta)*4*D[i]/dx[i] * (second[2*i+1] - second[2*i]) - (delta*2*D[i-1])/ dx[i-1]*(second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx[i]/2*source[i] + J_r - 2*D[i]/dx[i]    * (second[2*i+1] - second[2*i])

            if bcr == "ref":
                b_r = -D[i] / dx[i] * (second[2*i+1] - second[2*i])

        else:  # interior cell
            b_l = dx[i]/2*source[i]   + (1-delta)*4*D[i]/dx[i] * (second[2*i+1] - second[2*i]) - (delta*4*D[i-1])/dx[i-1]*(second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx[i]/2*source[i] - (1-delta)*4*D[i]/dx[i] * (second[2*i+1] - second[2*i]) + (delta*4*D[i+1])/dx[i+1]*(second[2*(i+1)+1]-second[2*(i+1)])

        b_vec[2 * i] = b_l
        b_vec[2 * i + 1] = b_r

    return b_vec


def compute_bound_rem(aflux, sflux, current):
    Rem_l = 0
    Rem_r = 0

    for m in range(N_angle):
        if angles[m] < 0:
            Rem_l += (
                weights[m]
                * angles[m]
                * (aflux[2 * m] + 0.5 * (sflux[0] + 3 * angles[m] * current[0]))
            )
        elif angles[m] > 0:
            Rem_r += (
                weights[m]
                * angles[m]
                * (
                    aflux[(N_cell - 1) * N_angle * 2 + 2 * m + 1]
                    + 0.5 * (sflux[-1] + 3 * angles[m] * current[-1])
                )
            )

    return (Rem_l, Rem_r)


#
# Second moment method!
#


# @nb.jit
def sec_mom_meth(aflux, af_edge, l):

    # number of interior cell edges
    N_edges = N_cell + 1

    # compute angular moments on the volumes and cells
    zeroth, first, second = compute_volume_moments(aflux)

    Rem_l, Rem_r = compute_bound_rem(aflux, zeroth, first)

    if SET_acc_debug:
        fig, axs = plt.subplots(1)
        axs[0].plot(np.linspace(0, N_cell, 2 * N_cell), zeroth, ".-", label="φ")
        axs[0].plot(np.linspace(0, N_cell, 2 * N_cell), first, label="J")
        axs[0].plot(np.linspace(0, N_cell, 2 * N_cell), second, "^", label="Γ")
        axs[0].set_xlabel("cell [j]")
        axs[0].set_ylabel("QOI Into DSA")
        axs[0].legend()
        plt.title(
            "INTO SMM SOLVE L={}\n S{}, σ={}, c={}, Δx={}".format(
                l, N_angle, sigma, sigma_s / sigma, dx
            )
        )
        plt.show()

    b_vec = build_smm_b(second)

    diff = build_smm_mat()

    zeroth_new = np.linalg.solve(diff, b_vec)

    # computing new current (first angular moment) (palmer eq 30/31)
    first_new = np.zeros_like(first)

    for i in range(N_cell):
        first_new[2 * i] = -1 / (sigma[i] * 3 * dx[i]) * (zeroth_new[2 * i + 1] - zeroth_new[2 * i]) - 4 / (sigma[i] * 3 * dx[i]) * (second[2 * i + 1] - second[2 * i])
        first_new[2 * i + 1] = -1 / (sigma[i] * 3 * dx[i]) * (zeroth_new[2 * i + 1] - zeroth_new[2 * i]) - 4 / (sigma[i] * 3 * dx[i]) * (second[2 * i + 1] - second[2 * i])

    # compute new updates with l+1/2 af, and new zeroth and first via yavuz clever update
    af_new_edge = np.zeros(N_edges * N_angle)

    # all volume terms used to update cell edges
    for i in range(N_edges):
        for m in range(N_angle):
            edge_index = (i) * N_angle + m
            if i == 0:
                af_new_edge[edge_index] = af_l
            elif i == N_cell:
                af_new_edge[edge_index] = af_r
            else:
                # af_i = 2*i*N_angle + 2*m
                if angles[m] > 0:
                    vol_index = 2 * (i - 1) + 1
                    vol_af_index = 2 * N_angle * (i - 1) + 2 * m + 1
                    af_new_edge[edge_index] = af_edge[edge_index] + (1 / 2) * (
                        (zeroth_new[vol_index] - zeroth[vol_index])
                        + 3 * angles[m] * (first_new[vol_index] - first[vol_index])
                    )
                elif angles[m] < 0:
                    vol_index = 2 * (i)
                    vol_af_index = 2 * N_angle * (i) + 2 * m
                    af_new_edge[edge_index] = af_edge[edge_index] + (1 / 2) * (
                        (zeroth_new[vol_index] - zeroth[vol_index])
                        + 3 * angles[m] * (first_new[vol_index] - first[vol_index])
                    )

    if SET_acc_debug:
        fig, axs = plt.subplots(1)

        axs[0].plot(np.linspace(0, N_cell, 2 * N_cell), zeroth_new, ".-", label="φ")
        axs[0].plot(np.linspace(0, N_cell, 2 * N_cell), first_new, label="J")
        axs[0].plot(np.linspace(0, N_cell, 2 * N_cell), second, "^", label="Γ")
        axs[0].set_xlabel("cell [j]")
        axs[0].set_ylabel("QOI OUT DSA")
        axs[0].legend()
        axs[0].set_title("OUT Volume averaged terms")
        plt.show()

    return (af_new_edge, second)


#
# Transport!
#


##@nb.jit
def transport():

    aflux_last = np.random.random(N)
    aflux_new = np.zeros(N)
    aflux = np.zeros(N)
    aflux_plottin = np.zeros(N)
    aflux_edge = np.zeros_like(aflux_last)

    aflux_edge = compute_aflux_edge(aflux_last)

    second_moments = np.zeros((N_cell * 2, 2))

    converged = False

    error = 1
    error_last = 1
    error2 = 1

    l = 0

    while not converged:

        # oci loop
        for i in range(N_cell):
            b = np.zeros(2 * N_angle)

            A = buildA(i)

            buildb(aflux_edge, b, i)

            aflux_cell = np.linalg.solve(A, b)

            for m in range(N_angle):
                aflux[i * N_angle * 2 + m * 2] = aflux_cell[m * 2]
                aflux[i * N_angle * 2 + m * 2 + 1] = aflux_cell[m * 2 + 1]

        aflux_edge = compute_aflux_edge(aflux)

        if SET_acc:
            aflux_edge, second = sec_mom_meth(aflux, aflux_edge, l)

        error = np.linalg.norm(aflux - aflux_last, ord=2)
        spec_rad = error / error_last

        if l > 1:
            if error < tol * (1 - spec_rad):
                converged = True
        if l >= max_it:
            converged = True
            print("warning: didn't converge after max iter")

        if printer:
            print("l ", l, " error ", error, " ρ ", spec_rad)

        error_last = error
        aflux_last[:] = aflux[:]
        l += 1

    zeroth, first, second = compute_volume_moments(aflux)

    return (zeroth, aflux, spec_rad, l)


#
# Actually run some problems
#


# Control plots and printings
SET_acc = True
SET_acc_debug = False
printer = True
af_printer = False

# Basic iteration parameters
max_it = int(4000)
tol = 1e-4


# Problem definition
Len = 4
dxv = 0.1
N_cell = int(Len / dxv)
dx = dxv * np.ones(N_cell)
N = 2 * N_angle * N_cell
x_edge = np.linspace(0, Len, 2 * N_cell + 1)  # edges between half cells
x_mid = 0.5 * (x_edge[:-1] + x_edge[1:])  # middle of half cells

# test problems
source_free_pure_absorber = False
infinite_pure_absorber = False
infinite_homogenous = False
regression_slab = False
slab_absorbium = False
yavuz_problem = True

#
# Source free pure absorber
#

if source_free_pure_absorber:

    source = np.zeros(N_cell)
    sigma = 4 * np.ones(N_cell)
    sigma_s = sigma * 0.0
    sigma_a = sigma - sigma_s
    D = 1 / (3 * sigma)

    af_l = 1
    af_r = 1

    J_r = af_r / 2
    J_l = af_l / 2

    zeroth_smm, aflux, spec_smm, l_smm = transport()

    SET_acc = False
    zeroth_oci, aflux, spec_oci, l_oci = transport()

    error_sol = np.linalg.norm(zeroth_smm - zeroth_oci, ord=2)

    plt.figure()
    plt.plot(x_mid, zeroth_oci, "k*", label="OCI (l={})".format(l_oci))
    plt.plot(x_mid, zeroth_smm, "m--", label="2nd (l={})".format(l_smm))
    plt.plot(x_mid, np.ones(x_mid.size)*af_l / (sigma_a[0]), ".r", label="Ref")
    plt.xlabel("x [cm]")
    plt.ylabel(r"$\phi$")
    plt.title(
        "Source Free, Pure Absorber, ε={:.3e}\n S{}, σ={}, c={}, Δx={}".format(
            error_sol, N_angle, sigma[0], sigma_s[0] / sigma[0], dx[0]
        )
    )
    plt.ylim([0, np.max(zeroth_smm) * 1.5])
    plt.legend()
    plt.grid()
    plt.show()


#
# Infinite homogenous pure absorber
#


if infinite_pure_absorber:

    source = np.ones(N_cell)
    sigma = 4 * np.ones(N_cell)
    sigma_s = sigma * 0.0
    sigma_a = sigma - sigma_s
    D = 1 / (3 * sigma)

    af_l = source[0] / (2 * sigma_a[0])
    af_r = source[-1] / (2 * sigma_a[0])

    J_r = af_r / 2
    J_l = af_l / 2

    zeroth_smm, aflux, spec_smm, l_smm = transport()

    SET_acc = False
    zeroth_oci, aflux, spec_oci, l_oci = transport()

    error_sol = np.linalg.norm(zeroth_smm - zeroth_oci, ord=2)

    plt.figure()
    plt.plot(x_mid, zeroth_oci, "k*", label="OCI (l={})".format(l_oci))
    plt.plot(x_mid, zeroth_smm, "m--", label="2nd (l={})".format(l_smm))
    plt.plot(x_mid, np.ones(x_mid.size)*source[0] / (sigma_a[0]), ".r", label="Ref")
    plt.xlabel("x [cm]")
    plt.ylabel(r"$\phi$")
    plt.title(
        "ထ-Pure Absorber, ε={:.3e}\n S{}, σ={}, c={}, Δx={}".format(
            error_sol, N_angle, sigma[0], sigma_s[0] / sigma[0], dx[0]
        )
    )
    plt.ylim([0, np.max(zeroth_smm) * 1.5])
    plt.legend()
    plt.grid()
    plt.show()


#
# Infinite homogenous slab
#


if infinite_homogenous:

    source = np.ones(N_cell)
    sigma = 4 * np.ones(N_cell)
    sigma_s = sigma * 0.9
    sigma_a = sigma - sigma_s
    D = 1 / (3 * sigma)

    af_l = source[0] / (2 * sigma_a[0])
    af_r = source[-1] / (2 * sigma_a[0])

    J_r = af_r / 2
    J_l = af_l / 2

    SET_acc = True
    zeroth_smm, aflux, spec_smm, l_smm = transport()

    SET_acc = False
    zeroth_oci, aflux, spec_oci, l_oci = transport()

    error_sol = np.linalg.norm(zeroth_smm - zeroth_oci, ord=2)

    plt.figure()
    plt.plot(x_mid, zeroth_oci, "k*", label="OCI (l={})".format(l_oci))
    plt.plot(x_mid, zeroth_smm, "m--", label="2nd (l={})".format(l_smm))
    plt.plot(x_mid, np.ones(x_mid.size)*source[0] / (sigma_a[0]), ".r", label="Ref")
    plt.xlabel("x [cm]")
    plt.ylabel(r"$\phi$")
    plt.title(
        "ထ-homo-medium, ε={:.3e}\n S{}, σ={}, c={}, Δx={}".format(
            error_sol, N_angle, sigma[0], sigma_s[0] / sigma[0], dx[0]
        )
    )
    plt.ylim([0, np.max(zeroth_smm) * 1.5])
    plt.legend()
    plt.grid()
    plt.show()


#
# Regression testing to LLD SI results from Palmer
#


if regression_slab:
    # comparing against lumped ld from Palmer

    Len = 4
    dxv = 0.05
    N_cell = int(Len / dxv)
    dx = dxv * np.ones(N_cell)
    N = 2 * N_angle * N_cell
    x_edge = np.linspace(0, Len, 2 * N_cell + 1)  # edges between half cells
    x_mid = 0.5 * (x_edge[:-1] + x_edge[1:])  # middle of half cells

    source = np.zeros(N_cell)
    sigma = 1.0 * np.ones(N_cell)
    sigma_s = sigma * 0.5
    sigma_a = sigma - sigma_s
    D = 1 / (3 * sigma)

    af_l = 5
    af_r = 0

    J_r = af_r / 2
    J_l = af_l / 2

    SET_acc = True
    zeroth_smm, aflux, spec_smm, l_smm = transport()

    SET_acc = False
    zeroth_oci, aflux, spec_oci, l_oci = transport()

    error_sol = np.linalg.norm(zeroth_smm - zeroth_oci, ord=2)

    from verification.lumped_ld_res import s2, s4, x_lldres

    plt.figure()
    plt.plot(x_mid, zeroth_oci, "k*", label="OCI (l={})".format(l_oci))
    plt.plot(x_mid, zeroth_smm, "m--", label="2nd (l={})".format(l_smm))
    if N_angle == 2:
        plt.plot(x_lldres, s2, ".r", label="Ref (Lumped LLD SI)")
    elif N_angle == 4:
        plt.plot(x_lldres, s4, ".r", label="Ref (Lumped LLD SI)")
    else:
        print("I don't have regression results for that many angles (using S4)")
        plt.plot(x_lldres, s4, ".r", label="Ref (Lumped LLD SI)")

    plt.xlabel("x [cm]")
    plt.ylabel(r"$\phi$")
    plt.title(
        "Regression test, ε={:.3e}\n S{}, σ={}, c={}, Δx={}".format(
            error_sol, N_angle, sigma[0], sigma_s[0] / sigma[0], dx[0]
        )
    )
    plt.ylim([0, np.max(zeroth_smm) * 1.15])
    plt.legend()
    plt.grid()
    plt.show()
    #plt.savefig("regression_slabs4.pdf")


#
# Slab absorbium (https://github.com/CEMeNT-PSAAP/MCDC/tree/dev/examples/fixed_source/slab_absorbium)
#

if slab_absorbium:
    # 3 region slab problem
    r1 = 1.5
    r2 = 2.0
    r3 = 1.0

    Len = 6
    dxv = 0.05
    N_cell = int(Len / dxv)
    dx = dxv * np.ones(N_cell)
    N = 2 * N_angle * N_cell
    x_edge = np.linspace(0, Len, 2 * N_cell + 1)  # edges between half cells
    x_mid = 0.5 * (x_edge[:-1] + x_edge[1:])  # middle of half cells

    source = np.ones(N_cell)

    sigma = r1 * np.ones(N_cell)
    sigma[int(N_cell / 3) : int(2 * N_cell / 3)] = r2
    sigma[int(2 * N_cell / 3) :] = r3

    sigma_s = sigma * 0.0
    sigma_a = sigma - sigma_s
    D = 1 / (3 * sigma)

    af_l = 0
    af_r = 0

    J_r = af_r / 2
    J_l = af_l / 2

    SET_acc = True
    zeroth_smm, aflux_smm, spec_smm, l_smm = transport()

    SET_acc = False
    zeroth_oci, aflux_oci, spec_oci, l_oci = transport()

    #error_sol = np.linalg.norm(zeroth_smm - zeroth_oci, ord=2)

    from verification.slab_absorbium_ref import reference

    zeroth_ref, first_ref, af_ref = reference(x_edge, angles)

    zeroth, first, second = compute_volume_moments(aflux_oci)

    # plotting flux
    plt.figure()
    plt.plot(x_mid, zeroth_oci, "k*", label="OCI {}".format(l_oci))
    plt.plot(x_mid, zeroth_smm, "m--", label="2nd {}".format(l_smm))
    plt.plot(x_mid, zeroth_ref, "r.", label="Ref.")
    plt.xlabel("x [cm]")
    plt.ylabel(r"$\phi$")
    plt.title("Slab Absorbium\n S{}, Δx={}".format(N_angle, dx[0]))
    plt.legend()
    plt.grid()
    #plt.savefig("slab_abs_sf.pdf")
    plt.show()

    R_l = 0
    for m in range(int(N_angle/2)):
        R_l += weights[m] * angles[m] * (aflux_oci[2*m] - .5*(zeroth[0] + 3*angles[m]*first[0]))




    #plt.figure()
    #plt.plot(x_mid, zeroth, "k*", label="φ".format(l_oci))
    #plt.plot(x_mid, first, "m--", label="J".format(l_smm))
    #plt.plot(x_mid, second, ".r", label="Γ")
    #plt.xlabel("distance [cm]")
    #plt.ylabel("QOI")
    #plt.title("Slab Absorbium, ε={:.3e}\n S{}, Δx={}".format(error_sol, N_angle, dx[0]))
    #plt.legend()
    #plt.grid()
    #plt.show()

    b_vec = build_smm_b(second)
    diff = build_smm_mat()
    zeroth_new = np.linalg.solve(diff, b_vec)

    #plt.figure()
    #plt.plot(x_mid, zeroth_oci, "k*", label="OCI (l={})".format(l_oci))
    #plt.plot(x_mid, zeroth_new, "m--", label="2nd (l={})".format(l_smm))
    #plt.plot(x_mid, zeroth_ref, ".r", label="Ref")
    #plt.xlabel("distance [cm]")
    #plt.ylabel("φ")
    #plt.title("Slab Absorbium, ε={:.3e}\n S{}, Δx={}".format(error_sol, N_angle, dx[0]))
    #plt.legend()
    #plt.grid()
    #plt.show()


    # Angular flux - spatial average
    vmin = min(np.min(af_ref), np.min(aflux_oci), np.min(aflux_smm))
    vmax = max(np.max(af_ref), np.max(aflux_oci), np.max(aflux_smm))
    fig, ax = plt.subplots(1, 3, sharey=True)

    Z, MU = np.meshgrid(x_mid, mu_mid)

    im = ax[0].contourf(MU.T, Z.T, af_ref, vmin=vmin, vmax=vmax, levels=6)
    ax[0].set_xlabel(r"Polar cosine, $\mu$")
    ax[0].set_ylabel(r"$x [cm]$")
    ax[0].set_title(r"Ref.")

    Z, MU = np.meshgrid(x_mid, angles)

    aflux_oci_sq = np.zeros((2*N_cell, N_angle))
    for i in range(N_cell):
        for m in range(N_angle):
            af_index = (i)*N_angle*2 + m*2
            aflux_oci_sq[2*i, m] = aflux_oci[af_index]
            aflux_oci_sq[2*i+1, m] = aflux_oci[af_index+1]

    ax[1].contourf(MU.T, Z.T, aflux_oci_sq, vmin=vmin, vmax=vmax, levels=6)
    ax[1].set_xlabel(r"Polar cosine, $\mu$")
    ax[1].set_title(r"OCI")

    aflux_smm_sq = np.zeros((2*N_cell, N_angle))
    for i in range(N_cell):
        for m in range(N_angle):
            af_index = (i)*N_angle*2 + m*2
            aflux_smm_sq[2*i, m] = aflux_smm[af_index]
            aflux_smm_sq[2*i+1, m] = aflux_smm[af_index+1]

    ax[2].contourf(MU.T, Z.T, aflux_smm_sq, vmin=vmin, vmax=vmax, levels=6)
    ax[2].set_xlabel(r"Polar cosine, $\mu$")
    ax[2].set_title(r"2nd")


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$\hat{\psi}$")
    plt.savefig("slab_abs_af.pdf")

    error = np.abs(aflux_smm_sq - aflux_oci_sq)

    fig, ax = plt.subplots()
    im = ax.contourf(MU.T, Z.T, error, levels=6)
    ax.set_xlabel(r"Polar cosine, $\mu$")
    ax.set_ylabel(r"$x$")
    ax.set_title(r"|$\psi_{oci}-\psi_{smm}$|")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    plt.show()



# problem from Yavuz and Larsen comparing to Monte Carlo results
if yavuz_problem:

    Len = 4
    dxv = 0.01
    N_cell = int(Len / dxv)
    dx = dxv * np.ones(N_cell)
    N = 2 * N_angle * N_cell
    x_edge = np.linspace(0, Len, 2 * N_cell + 1)  # edges between half cells
    x_mid = 0.5 * (x_edge[:-1] + x_edge[1:])  # middle of half cells

    sigma_s = np.ones(N_cell)
    sigma_s[0:int(N_cell/4)] = 1.0
    sigma_s[int(N_cell/4):int(N_cell/2)] = 0.95
    sigma_s[int(N_cell/2):int(3*N_cell/4)] = 0.8
    sigma_s[int(3*N_cell/4):] = 0.95

    sigma = np.ones(N_cell)

    source = np.zeros(2*N_cell)
    source[int(N_cell/4):int(N_cell/2)] = 1.0
    source[int(N_cell/2):int(3*N_cell/4)] = 1.0

    sigma_a = sigma - sigma_s
    D = 1 / (3 * sigma)

    af_l = 0
    af_r = 0

    J_r = af_r / 2
    J_l = af_l / 2

    SET_acc = True
    zeroth_smm, aflux_smm, spec_smm, l_smm = transport()

    SET_acc = False
    zeroth_oci, aflux_oci, spec_oci, l_oci = transport()


    # plotting flux
    plt.figure()
    plt.plot(x_mid, zeroth_oci, "k*", label="oci".format(l_oci))
    plt.plot(x_mid, zeroth_smm, "m--", label="smm".format(l_smm))
    plt.xlabel("x [cm]")
    plt.ylabel(r"$\phi$")
    plt.title("Yavuz Problem\n S{}, Δx={}".format(N_angle, dx[0]))
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig("yavuz_fluxes.pdf")

    # Load Monte Carlo results
    import h5py
    with h5py.File("verification/yavuz_problem.h5", "r") as f:
        # space
        z = f["tallies/mesh_tally_0/grid/z"][:]
        dz = z[1:] - z[:-1]
        z_mid = 0.5 * (z[:-1] + z[1:])

        # angles
        mu = f["tallies/mesh_tally_0/grid/mu"][:]
        dmu = mu[1:] - mu[:-1]
        mu_mid = 0.5 * (mu[:-1] + mu[1:])

        # angular flux and staistical error
        af_ref = f["tallies/mesh_tally_0/flux/mean"][:]
        af_ref = np.transpose(af_ref)
        psi_sd = f["tallies/mesh_tally_0/flux/sdev"][:]
    I = len(z) - 1
    N = len(mu) - 1

    print("")
    print("")
    print(r"Monte Carlo Camprison Result Varaince (sum(σ)): ", np.sum(psi_sd))
    print("")
    print("")

    # Scalar flux
    zeroth_mc = np.zeros(I)
    for i in range(I):
        zeroth_mc[i] += np.sum(af_ref[i, :])

    # Normalize
    zeroth_mc /= dz
    for n in range(N):
        af_ref[:, n] = af_ref[:, n] / dz / dmu[n]

    # sf /= max(sf)
    zeroth_oci /= np.max(zeroth_oci)
    zeroth_smm /= np.max(zeroth_smm)
    zeroth_mc /= np.max(zeroth_mc)

    # plotting normalized flux
    plt.figure()
    plt.plot(x_mid, zeroth_oci, "k*", label="OCI ({})".format(l_oci))
    plt.plot(x_mid, zeroth_smm, "m--", label="2nd ({})".format(l_smm))
    plt.plot(z_mid, zeroth_mc, ".r", label="Ref.")
    plt.xlabel("x [cm]")
    plt.ylabel(r"$\phi / |\phi|_{max}$")
    plt.text(0.5, 0.25, r"$|\sigma_{MC}|_1 = $ %.2e"%np.sum(psi_sd))
    #plt.title("Yavuz Problem\n S{}, Δx={}".format(N_angle, dx[0]))
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig("yavuz_normalized_fluxes.pdf")

    # changing shape of matercies for plotting
    aflux_oci_sq = np.zeros((2*N_cell, N_angle))
    aflux_smm_sq = np.zeros((2*N_cell, N_angle))
    for i in range(N_cell):
        for m in range(N_angle):
            af_index = (i)*N_angle*2 + m*2
            aflux_oci_sq[2*i, m] = aflux_oci[af_index]
            aflux_oci_sq[2*i+1, m] = aflux_oci[af_index+1]
            aflux_smm_sq[2*i, m] = aflux_smm[af_index]
            aflux_smm_sq[2*i+1, m] = aflux_smm[af_index+1]
    
    # af /= max(af)
    aflux_oci_sq /= np.max(aflux_oci_sq)
    aflux_smm_sq /= np.max(aflux_smm_sq)
    af_ref /= np.max(af_ref)

    # Angular flux - spatial average
    vmin = 0.0 #min(np.min(af_ref), np.min(aflux_oci), np.min(aflux_smm))
    vmax = 1.0 # max(np.max(af_ref), np.max(aflux_oci), np.max(aflux_smm))
    print(vmin)
    print(vmax)

    fig, ax = plt.subplots(1, 3, sharey=True)

    Z, MU = np.meshgrid(z_mid, mu_mid)
    im = ax[0].contourf(MU.T, Z.T, af_ref, vmin=vmin, vmax=vmax, levels=6)
    ax[0].set_xlabel(r"$\mu$")
    ax[0].set_ylabel(r"$x$")
    ax[0].set_title(r"Ref. (MC)")

    # need to remake for OCI descritizations
    Z, MU = np.meshgrid(x_mid, angles)

    ax[1].contourf(MU.T, Z.T, aflux_oci_sq, vmin=vmin, vmax=vmax, levels=6)
    ax[1].set_xlabel(r"$\mu$")
    ax[1].set_title(r"OCI")

    ax[2].contourf(MU.T, Z.T, aflux_smm_sq, vmin=vmin, vmax=vmax, levels=6)
    ax[2].set_xlabel(r"$\mu$")
    ax[2].set_title(r"2nd")

    # color bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$\psi/|\psi|_{max}$")
    #plt.show()
    plt.savefig("yavuz_af.pdf")