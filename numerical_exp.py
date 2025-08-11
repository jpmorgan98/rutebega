import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import numba as nb

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

# [1] M. Yavuz and E. Larsen (1989) *Spatial domain decomposition for Neutron transport problems* Transport Theory and Statistical Physics 18:2, 205-219 
# [2] T. Palmer Private corispondence

#N_cell = 100
N_angle = 8
dx = 1.0

Len = 25
N_cell = int(Len/dx)

N = 2*N_cell*N_angle

sigma = 1.5
sigma_s = 0*sigma
sigma_a = sigma - sigma_s

D = 1/(3*sigma)

tol = 1e-5
max_it = int(4000)

SET_palmer_acc = True
SET_palmer_debug = False
printer = True
af_printer = False

[angles, weights] = np.polynomial.legendre.leggauss(N_angle)

source = 0.0

af_l = 0 #source/(2*sigma_a)
af_r = 0 #source/(2*sigma_a)

sf_l = 0
sf_r = 0

J_l = 0
J_r = 0

Rem_l = 0
Rem_r = 0

bcl = "in"
bcr = "in"

bcl_d = "yavuz"
bcr_d = "yavuz"

second_l = 0
second_r = 0

#@nb.jit
def Sbuild():
    S = np.zeros((2*N_angle,2*N_angle))
    #for m in range(N_angle):

    beta = sigma_s * dx / 4

    for p in range(N_angle):
        for j in range(N_angle):
            S[p*2,   j*2]   = beta*weights[j]
            S[p*2+1, j*2+1] = beta*weights[j]

    return(S)

#@nb.jit
def Ablock(mu):
    return(
        np.array(((np.abs(mu)/2 + sigma*dx/2, mu/2),
                  (-mu/2,                     np.abs(mu)/2 + sigma*dx/2)))
    )

#@nb.jit
def buildA():
    A = np.zeros((2*N_angle, 2*N_angle))
    for a in range(N_angle):
        A[a*2:(a+1)*2,a*2:(a+1)*2] = Ablock(angles[a])

    S = Sbuild()

    A.shape == S.shape

    A = A-S

    return(A)

#@nb.jit
def b_neg(psi_rightBound, mu):
    b_n = np.array((dx/4*source,
                    dx/4*source - mu*psi_rightBound))
    return(b_n)

#@nb.jit
def b_pos(psi_leftBound, mu):
    b_p = np.array((dx/4*source + mu*psi_leftBound,
                    dx/4*source))
    return(b_p)

#@nb.jit
def buildb(aflux_last, b, i):

    for m in range(N_angle):
        if angles[m]>0:
            if (i==0):
                if (bcl == "in"):
                    af_lb = af_l
                elif (bcl == "ref"):
                    ref_angle_ord = int(N_angle-m-1)
                    af_lb = aflux_last[0 + ref_angle_ord*2 + 0]
            else:
                af_lb = aflux_last[2*N_angle*(i-1) + m*2 + 1]

            b[m*2:(m+1)*2] = b_pos(af_lb, angles[m])

        elif angles[m]<0:
            if (i==N_cell-1):
                if (bcr == "in"):
                    af_rb = af_r
                elif (bcr == "ref"):
                    ref_angle_ord = int(N_angle-m-1)  
                    af_rb = aflux_last[2*N_angle*(i-1) + ref_angle_ord*2 + 1]
            else:
                af_rb = aflux_last[2*N_angle*(i+1) + m*2 + 0]

            b[m*2:(m+1)*2] = b_neg(af_rb, angles[m])


#@nb.jit
def buildb2(aflux_edge, b, i):

    i = int(i)

    for m in range(N_angle):

        af_lb = aflux_edge[i*N_angle+m]
        if angles[m]>0:
            af_lb = aflux_edge[i*N_angle + m]

            b[m*2:(m+1)*2] = b_pos(af_lb, angles[m])

        elif angles[m]<0:
            af_rb = aflux_edge[(i+1)*N_angle + m]

            b[m*2:(m+1)*2] = b_neg(af_rb, angles[m])

#@nb.jit
def compute_aflux_edge(aflux):
    # explicit implementation of upstream closure

    N_edges = int(N_cell+1)
    af_edge = np.zeros(N_edges*N_angle)

    for i in range(N_edges):
        if i == 0:
            for m in range(N_angle):
                if bcl == 'in':
                    af_edge[i*N_angle + m] = af_l
                elif bcl == 'ref':
                    ref_angle_ord = int(N_angle-m-1)
                    af_index = (0)*N_angle*2 + ref_angle_ord*2 + 1 
                    af_edge[(i)*N_angle + m] = aflux[af_index]
                else:
                    print("Check bcl")

        elif i == N_cell:
            for m in range(N_angle):
                if bcr == 'in':
                    af_edge[(i)*N_angle + m] = af_r
                elif bcr == 'ref':
                    ref_angle_ord = int(N_angle-m-1)
                    af_index = (i-1)*N_angle*2 + ref_angle_ord*2 + 1 
                    af_edge[(i)*N_angle + m] = aflux[af_index]
                else:
                    print("Check bcr")

        else:
            for m in range(N_angle):

                if angles[m] > 0: 
                    af_index = (i-1)*N_angle*2 + m*2 + 1 
                    af_edge[(i)*N_angle+ m] = aflux[af_index]

                elif angles[m] < 0: 

                    af_index = (i)*N_angle*2 + m*2
                    af_edge[(i)*N_angle + m] = aflux[af_index]

    return(af_edge)

#@nb.jit
def compute_moments_val(af):
    zeroth = 0
    first  = 0
    second = 0

    for m in range(N_angle):
        zeroth += weights[m] * af
        first  += weights[m] * angles[m] * af
        second += weights[m] * (.5*(3*angles[m]**2 - 1)) * af

    return(zeroth, first, second)


def compute_partial_current(af, dir):
    first  = 0
    second = 0

    for m in range(N_angle):
        if (dir=='p' and angles[m]>0):
            first  += weights[m] * angles[m] * af
            second += weights[m] * (.5*(3*angles[m]**2 - 1)) * af
        elif(dir=='n' and angles[m]<0):
            first  -= weights[m] * angles[m] * af
            second += weights[m] * (.5*(3*angles[m]**2 - 1)) * af

    return(first, second)


def compute_partial_current_cont(af, dir):
    first  = 0

    for m in range(N_angle):
        if (dir=='p' and angles[m]>0):
            first  += weights[m] * angles[m] * af
        elif(dir=='n' and angles[m]<0):
            first  -= weights[m] * angles[m] * af

    return(first)

def partial_current_bc_reflect(af, dir):
    first  = 0

    for m in range(N_angle):
        if (dir=='l' and angles[m]>0):
            first  += weights[m] * angles[m] * af[2*(N_angle-1-m)]
        elif(dir=='r' and angles[m]<0):
            first  -= weights[m] * angles[m] * af[2*N_angle*(N_cell-1) + 2*(N_angle-1-m) + 1]

    return(first)

def partial_current_bc(af, dir):
    first  = 0

    for m in range(N_angle):
        if (dir=='l' and angles[m]>0):
            first  += weights[m] * angles[m] * af[2*(m)]
        elif(dir=='r' and angles[m]<0):
            first  -= weights[m] * angles[m] * af[2*N_angle*(N_cell-1) + 2*(m) + 1]

    return(first)

#@nb.jit
def compute_volume_moments(af):

    N_mom = 2*N_cell

    zeroth = np.zeros(N_mom)
    first  = np.zeros(N_mom)
    second = np.zeros((N_mom))

    for j in range(N_cell):
        for m in range(N_angle):
            af_index = j*N_angle*2 + m*2

            # left
            zeroth[2*j]   += weights[m] * af[af_index]
            first [2*j]   += weights[m] * angles[m] * af[af_index]
            second[2*j]   += weights[m] * (.5*(3*(angles[m]**2) - 1)) * af[af_index] #(.5*(3*angles[m]**2 - 1))

            # right
            zeroth[2*j+1] += weights[m] * af[af_index + 1]
            first [2*j+1] += weights[m] * angles[m] * af[af_index + 1]
            second[2*j+1] += weights[m] * (.5*(3*(angles[m]**2) - 1)) * af[af_index + 1] #(.5*(3*(angles[m]**2) - 1))

    return(zeroth, first, second)



def compute_cell_edge_moments(af):

    """
          j_j+1/2
            |
          ->|<-
            |
   +=wμψ_jr | +=wμψ_j+1l
    """

    # number of total edges
    N_edges = N_cell+1
    # the solution on the boundaries is known! Except for reflecting

    edge_sf = np.zeros((N_edges))
    edge_current = np.zeros((N_edges))
    edge_second = np.zeros((N_edges))

    for j in range(N_edges):
        for m in range(N_angle):
            
            if j ==0:
                edge_sf[j]      = sf_l
                edge_current[j] = J_l
                edge_second[j]  = second_l
            elif j == N_cell:
                edge_sf[j]      = sf_r
                edge_current[j] = J_r
                edge_second[j]  = second_r
            else:
                if angles[m] > 0:
                    af_index = (j-1)*N_angle*2 + m*2 + 1
                    edge_sf[j]      += weights[m] * af[af_index]
                    edge_current[j] += weights[m] * angles[m] * af[af_index]
                    edge_second[j]  += weights[m] * (.5*(3*(angles[m]**2) - 1)) * af[af_index] # (.5*(3*(angles[m]**2) - 1))
                elif angles[m] < 0:
                    af_index = (j)*N_angle*2 + m*2
                    edge_sf[j]      += weights[m] * af[af_index]
                    edge_current[j] += weights[m] * angles[m] * af[af_index]
                    edge_second[j]  += weights[m] *  (.5*(3*(angles[m]**2) - 1)) * af[af_index] #
                

    return(edge_sf, edge_current, edge_second)


#@nb.jit
def build_diffusion_mat():
    # building the diffusion A mat
    # this system is *very* sparse, doing this as dense is silly but ok for now

    delta = 1/2
    gamma = 1/4

    A = np.zeros((2*N_cell, 2*N_cell))

    for i in range(N_cell):

        A[2*i,2*i]   = (1-delta)*D/dx + gamma + sigma_a*dx/2
        A[2*i,2*i+1] = -(1-delta)*D/dx

        A[2*i+1,2*i]   = -(1-delta)*D/dx 
        A[2*i+1,2*i+1] = (1-delta)*D/dx + gamma + sigma_a*dx/2

        if (i==N_cell-1): #right bound (no right of cell info)
            A[2*i,2*(i-1)]   = -delta*D/dx
            A[2*i,2*(i-1)+1] = delta*D/dx - gamma

            # overwriting the last row
            if bcr == 'ref':
                A[2*i+1,2*i] = -(delta)*D/dx 
                A[2*i+1,2*i+1] = (delta)*D/dx + sigma_a*dx/2

        elif (i==0): #left bound (no left of cell info)

            A[2*i+1,2*(i+1)]   = -gamma+delta*D/dx
            A[2*i+1,2*(i+1)+1] = -delta*D/dx

            # over writing the first row
            if bcl == 'ref':
                A[2*i,2*i] = (delta)*D/dx + sigma_a*dx/2
                A[2*i,2*i+1] = -(delta)*D/dx 
            
            #elif bcl == "in":
            #    A[2*i,2*i]   = delta + D/dx + sigma_a*dx/2
            #    A[2*i,2*i+1] = -D/dx

        else: #interior cell
            #cell j-1 info
            A[2*i,2*(i-1)]   = -delta*D/dx
            A[2*i,2*(i-1)+1] = delta*D/dx - gamma
            # cell j+1 info=
            A[2*i+1,2*(i+1)]   = -gamma+delta*D/dx
            A[2*i+1,2*(i+1)+1] = -delta*D/dx

    return(A)

#@nb.jit
def diff_b_simple_revised(second):
    delta = 0.5

    b_vec = np.zeros(2*N_cell)
    for i in range(N_cell):
        if i == 0: # left hand bound, using bcs
            b_l = dx/2*source + (J_l) + 2*D/dx*(second[2*i+1]-second[2*i]) 
            b_r = dx/2*source - (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) + (delta*4*D)/dx * (second[2*(i+1)+1]-second[2*(i+1)])

        elif i==N_cell-1: #right hand bcs
            b_l = dx/2*source + (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (delta*4*D)/dx * (second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx/2*source + J_r - 2*D/dx*(second[2*i+1]-second[2*i]) #- 

            if bcr == 'ref':
                b_r = - D/dx*(second[2*i+1]-second[2*i])

        else: # interior cell
            b_l = dx/2*source + (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (delta*4*D)/dx * (second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx/2*source - (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) + (delta*4*D)/dx * (second[2*(i+1)+1]-second[2*(i+1)])

        b_vec[2*i]   = b_l
        b_vec[2*i+1] = b_r

    return(b_vec)


def diff_b_simple_palmer_bounds(second, Rem_l, Rem_r):
    delta = 0.5

    b_vec = np.zeros(2*N_cell)
    for i in range(N_cell):
        if i == 0: # left hand bound, using bcs
            b_l = dx/2*source + (J_l) + Rem_l + D/dx*(second[2*i+1]-second[2*i]) 
            b_r = dx/2*source - (delta-1)*2*D/dx * (second[2*i+1]-second[2*i]) + (delta*2*D)/dx * (second[2*(i+1)+1]-second[2*(i+1)])

            if bcr == 'ref':
                b_l = - D/dx*(second[2*i+1]-second[2*i])

        elif i==N_cell-1: #right hand bcs
            b_l = dx/2*source + (1-delta)*2*D/dx * (second[2*i+1]-second[2*i]) - (delta*2*D)/dx * (second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx/2*source + J_r - D/dx*(second[2*i+1]-second[2*i]) #- 

            if bcr == 'ref':
                b_r = - D/dx*(second[2*i+1]-second[2*i])

        else: # interior cell
            b_l = dx/2*source + (1-delta)*2*D/dx * (second[2*i+1]-second[2*i]) - (delta*2*D)/dx * (second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx/2*source - (delta-1)*2*D/dx * (second[2*i+1]-second[2*i]) + (delta*2*D)/dx * (second[2*(i+1)+1]-second[2*(i+1)])

        b_vec[2*i]   = b_l
        b_vec[2*i+1] = b_r

    return(b_vec)


def diff_b_full(second, second_edge):
    delta = 0.5
    D = 1/(3*sigma)

    b_vec = np.zeros(2*N_cell)

    for i in range(N_cell):

        avg_in_cell = (second[2*i+1]+second[2*i])/2
        
        if i == 0: # left hand bound, using bcs
            b_l = dx/2*source + J_l + 2*D/dx*(second_l - avg_in_cell)

            b_r = dx/2*source - 2*D/dx * (second_edge[i+1]-second_edge[i]) \
                  + delta*( 4*D/dx*(second_edge[i+1] - avg_in_cell) \
                           +4*D/dx*((second[2*(i+1)+1]+second[2*(i+1)])/2 - second_edge[i+1])) 

        elif i==N_cell-1: #right hand bcs
            b_l = dx/2*source + 2*D/dx * (second_edge[i+1]-second_edge[i]) \
            - delta*( 4*D/dx*( avg_in_cell -  second_edge[i]) \
                     +4*D/dx*( second_edge[i] - (second[2*(i-1)+1]+second[2*(i-1)])/2 ))
            
            b_r = dx/2*source + J_r - 2*D/dx*(avg_in_cell - second_edge[i+1])

        else: # interior cell
            b_l = dx/2*source + 2*D/dx*(second_edge[i+1]-second_edge[i]) \
            - delta*( 4*D/dx*( avg_in_cell -  second_edge[i]) \
                     +4*D/dx*( second_edge[i] - (second[2*(i-1)+1]+second[2*(i-1)])/2 ))

            b_r = dx/2*source - 2*D/dx*(second_edge[i+1]-second_edge[i]) \
                  + delta*( 4*D/dx*(second_edge[i+1] - avg_in_cell) \
                           +4*D/dx*((second[2*(i+1)+1]+second[2*(i+1)])/2 - second_edge[i+1])) 

        b_vec[2*i]   = b_l
        b_vec[2*i+1] = b_r

    return(b_vec)


def compute_bound_rem(aflux, sflux, current):
    Rem_l = 0
    Rem_r = 0

    for m in range(N_angle):
        if angles[m] < 0:
            Rem_l += weights[m] * angles[m] * (aflux[2*m] + .5*(sflux[0] + 3*angles[m]*current[0]))
        elif angles[m] > 0:
            Rem_r += weights[m] * angles[m] * (aflux[(N_cell-1)*N_angle*2 + 2*m + 1] + .5*(sflux[-1] + 3*angles[m]*current[-1]))

    return(Rem_l, Rem_r)

# dsa, full_smm, simple_smm
accelerator = 'simple_smm'

#@nb.jit
def palmer_acc(aflux, af_edge, l):
    # using moment description of current and scalar flux as that makes sense in a second moment method!

    # number of interior cell edges
    N_edges = N_cell+1

    # compute angular moments on the volumes and cells
    zeroth, first, second = compute_volume_moments(aflux)
    zeroth_edge, first_edge, second_edge = compute_cell_edge_moments(aflux)

    Rem_l, Rem_r = compute_bound_rem(aflux, zeroth, first)

    if accelerator == 'dsa':
        second *= 0
        second_edge *= 0

    #if (SET_palmer_debug):
    #    fig,axs = plt.subplots(2)
    #    axs[0].plot(np.linspace(0, N_cell, 2*N_cell), zeroth, '.-',label='φ')
    #    axs[0].plot(np.linspace(0, N_cell, 2*N_cell), first, label='J')
    #    axs[0].plot(np.linspace(0, N_cell, 2*N_cell), second, '^', label='Γ')
    #    axs[0].set_xlabel("cell [j]")
    #    axs[0].set_ylabel("QOI Into DSA")
    #    axs[0].legend()
    #    axs[0].set_title("INTO Volume averaged terms")
    #    axs[1].plot(np.linspace(0, N_cell, N_edges), zeroth_edge, '.-', label='φ')
    #    axs[1].plot(np.linspace(0, N_cell, N_edges), first_edge, label='J')
    #    axs[1].plot(np.linspace(0, N_cell, N_edges), second_edge, '^', label='Γ')
    #    axs[1].set_xlabel("cell [j]")
    #    axs[1].set_ylabel("QOI Into DSA")
    #    axs[1].legend()
    #    axs[1].set_title("Cell edge")
    #    #plt.title("L={}\n S{}, σ={}, c={}, Δx={}".format(l, N_angle, sigma, sigma_s/sigma, dx))
    #    plt.show()
    #
    delta = 0.5
    gamma = 0.25

    if accelerator == 'full_smm':
        b_vec = diff_b_full(second, second_edge)
    else:
        b_vec = diff_b_simple_revised(second) #diff_b_simple_revised(second)
    
    #b_vec = diff_b_simple_palmer_bounds(second, Rem_l, Rem_r)
    #b_vec = diff_b_simple_revised(second)
    #print("full ssm vector\n",diff_b_full(second, second_edge))
    #print("small ssm vector\n",diff_b_simple(second))

    diff = build_diffusion_mat()

    #print(D)
    #print(sigma)
    #print(dx)
    #print("second edge\n", second_edge)
    #print("second\n", second)
    #print(b_vec)
    #print()
    #print(diff)

    zeroth_new = np.linalg.solve(diff, b_vec)
    
    #if SET_palmer_debug:
    #
    #    zeroth_new = np.linalg.solve(diff, diff_b_simple_revised(second))
    #    #zeroth_new_full = np.linalg.solve(diff, diff_b_full(second, second_edge))
    #    zeroth_new_dsa = np.linalg.solve(diff, diff_b_simple_revised(np.zeros_like(second)))
    #    plt.figure()
    #    plt.plot(np.linspace(0, N_cell, 2*N_cell), zeroth_new, '.-', label="revised")
    #    #plt.plot(np.linspace(0, N_cell, 2*N_cell), zeroth_new_full, '*-', label="full")
    #    plt.plot(np.linspace(0, N_cell, 2*N_cell), zeroth_new_dsa, '--', label="dsa")
    #    plt.title("ACCELERATION SOLUTIONS AT CONVERGENCE")
    #    plt.xlabel("Distance")
    #    plt.ylabel("φ")
    #    plt.legend()
    #    plt.show()

    # computing new current (first angular moment) (palmer eq 30/31)

    #print(second_edge)
    #print(second)

    first_new = np.zeros_like(first)

    #print(zeroth_new.shape)
    #print(second_edge.shape)
    #print(second.shape)
    #print(first_new.shape)
    
    if accelerator == 'full_smm':
    #first_new_edge = np.zeros_like(first_edge)
        for i in range(N_cell):
            #second_avg = (second[2*i+1]+second[2*i])/2
            if i == 0:
                first_new[2*i] = first_edge[0]
                first_new[2*i+1] = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second_edge[i+1] - second_avg)
            elif i == N_cell-1:
                first_new[2*i]   = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second_avg - second_edge[i])
                first_new[2*i+1] = J_r
            else:
                first_new[2*i]   = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second_avg - second_edge[i])
                first_new[2*i+1] = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second_edge[i+1] - second_avg)
    else:
        for i in range(N_cell):
            if i == 0:
                first_new[2*i]   = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second[2*i+1]-second[2*i])
                first_new[2*i+1] = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second[2*i+1]-second[2*i])
            elif i == N_cell-1:
                first_new[2*i]   = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second[2*i+1]-second[2*i])
                first_new[2*i+1] = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second[2*i+1]-second[2*i])
            else:
                first_new[2*i]   = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second[2*i+1]-second[2*i])
                first_new[2*i+1] = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second[2*i+1]-second[2*i]) 
            #first_new[2*i]   = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second[2*i+1]-second[2*i])
            #first_new[2*i+1] = -1/(sigma*3*dx)*(zeroth_new[2*i+1]-zeroth_new[2*i]) - 4/(sigma*3*dx)*(second[2*i+1]-second[2*i]) 
    
    #zeroth_new_edge = np.zeros(N_edges)
    first_new_edge = np.zeros(N_edges)
    zeroth_new_edge = np.zeros(N_edges)

    #for i in range(N_edges):
        #if i ==0:
        #    first_new_edge[i] = J_l
        #    zeroth_new_edge[i] = sf_l
        #elif i ==N_cell:
        #    first_new_edge[i] = J_r
        #    zeroth_new_edge[i] = sf_r
        #else:
        #first_new_edge[i] =  .25*(zeroth_new[2*(i-1)+1]-zeroth_new[2*(i)]) + .5*(first_new[2*(i-1)+1]+first_new[2*(i)]) 
        #zeroth_new_edge[i] = (zeroth_new[2*(i-1)+1]+zeroth_new[2*(i)]) / 2

    # compute new updates with l+1/2 af, and new zeroth and first via yavuz clever update
    af_new_edge = np.zeros(N_edges*N_angle)
    aflux_new = np.zeros_like(aflux)
    #aflux_new = np.zeros_like(aflux)
    correction = np.zeros_like(aflux)

    
    """        Cellwise diagram
    0    j-1    1    j      2     j+1   3
    |           |           |           |
    |     /     |     /     |     /     |
    |   L / R   |   L / R   |   L / R   |
    |     /     |     /     |     /     |
    |           |           |           |
  j-(1+1/2)   j-1/2       j+1/2      j+(1+1/2)

    index i and N of edge values are boundary conditions!
    so when indexing edge values
    """

    """
    # all cell edge terms to update edges
    for i in range(N_edges):
        for m in range(N_angle):
            if i == 0:
                af_new_edge[(i)*N_angle + m] = af_l
            elif i == N_cell:
                af_new_edge[(i)*N_angle + m] = af_r
            else:
                edge_index = (i)*N_angle+ m
                if angles[m] > 0:
                    vol_index = 2*(i-1) + 1
                    af_new_edge[edge_index] = af_edge[edge_index] + (1/2)*((zeroth_new[vol_index] - zeroth[vol_index]) + 3*angles[m]*(first_new_edge[i] - first_edge[i]))
                elif angles[m] < 0:
                    vol_index = 2*(i)
                    af_new_edge[edge_index] = af_edge[edge_index] + (1/2)*((zeroth_new[vol_index] - zeroth[vol_index]) + 3*angles[m]*(first_new_edge[i] - first_edge[i]))
    """

    # all volume terms used to update cell edges
    for i in range(N_edges):
        for m in range(N_angle):
            edge_index = (i)*N_angle+ m
            if i == 0:
                af_new_edge[edge_index] = af_l
            elif i == N_cell:
                af_new_edge[edge_index] = af_r
            else:
            #af_i = 2*i*N_angle + 2*m
                if angles[m] > 0:
                    vol_index = 2*(i-1) + 1
                    vol_af_index = 2*N_angle*(i-1) + 2*m + 1
                    af_new_edge[edge_index] = af_edge[edge_index] + (1/2)*((zeroth_new[vol_index] - zeroth[vol_index]) + 3*angles[m]*(first_new[vol_index] - first[vol_index]))
                elif angles[m] < 0:
                    vol_index = 2*(i)
                    vol_af_index = 2*N_angle*(i) + 2*m
                    af_new_edge[edge_index] = af_edge[edge_index] + (1/2)*((zeroth_new[vol_index] - zeroth[vol_index]) + 3*angles[m]*(first_new[vol_index] - first[vol_index]))
    
    #for i in range(N_cell):
    #    for m in range(N_angle):
    #        af_i = 2*i*N_angle + 2*m
    #        aflux_new[af_i]   = aflux[af_i]   + (1/2)*((zeroth_new[2*i]   - zeroth[2*i])   + 3*angles[m]*(first_new[2*i]   - first[2*i]))
    #        aflux_new[af_i+1] = aflux[af_i+1] + (1/2)*((zeroth_new[2*i+1] - zeroth[2*i+1]) + 3*angles[m]*(first_new[2*i+1] - first[2*i+1]))
    #print(np.linalg.norm(correction))
    
    #if SET_palmer_debug:
    #    fig,axs = plt.subplots(2)
#
    #    axs[0].plot(np.linspace(0, N_cell, 2*N_cell), zeroth_new, '.-',label='φ')
    #    axs[0].plot(np.linspace(0, N_cell, 2*N_cell), first_new, label='J')
    #    axs[0].plot(np.linspace(0, N_cell, 2*N_cell), second, '^', label='Γ')
    #    axs[0].set_xlabel("cell [j]")
    #    axs[0].set_ylabel("QOI OUT DSA")
    #    axs[0].legend()
    #    axs[0].set_title("OUT Volume averaged terms")
#
    #    axs[1].plot(np.linspace(0, N_cell, N_edges), zeroth_new_edge, '.-', label='φ')
    #    axs[1].plot(np.linspace(0, N_cell, N_edges), first_new_edge, label='J')
    #    axs[1].plot(np.linspace(0, N_cell, N_edges), second_edge, '^', label='Γ')
    #    axs[1].set_xlabel("cell [j]")
    #    axs[1].set_ylabel("QOI OUT DSA")
    #    axs[1].legend()
    #    axs[1].set_title("Cell edge")
    #    plt.show()

    return(af_new_edge, second)

##@nb.jit
def transport():

    
    aflux_last = np.random.random(N)
    aflux_new = np.zeros(N)
    aflux = np.zeros(N)
    aflux_plottin = np.zeros(N)
    aflux_edge = np.zeros_like(aflux_last)

    aflux_edge = compute_aflux_edge(aflux_last)

    second_moments = np.zeros((N_cell*2,2))

    converged = False

    error = 1
    error_last = 1
    error2 = 1

    l = 0

    spec_rad_list = []

    while (not converged):

        #oci loop
        for i in range(N_cell):
            b = np.zeros(2*N_angle)

            A = buildA()

            buildb2(aflux_edge, b, i)
            #buildb(aflux_last, b, i)

            aflux_cell = np.linalg.solve(A, b)

            for m in range(N_angle):
                aflux[i*N_angle*2 + m*2]     = aflux_cell[m*2]
                aflux[i*N_angle*2 + m*2 + 1] = aflux_cell[m*2 + 1]

        aflux_edge = compute_aflux_edge(aflux)

        if (SET_palmer_acc):
            aflux_edge, second = palmer_acc(aflux, aflux_edge, l)
            #second_moments = np.append(second_moments, second, axis=1)
            
        #    error = np.linalg.norm(aflux_plottin-aflux_last, ord=2)
        #else:
        error = np.linalg.norm(aflux-aflux_last, ord=2)
        spec_rad = error/error_last
        if l>1:
            spec_rad_list.append(spec_rad)

        if l>1:
            if (error<tol*(1-spec_rad)):
                converged = True
        if l>=max_it:
            converged = True
            print("warning: didn't converge after max iter")

        if (printer):
            print("l ",l," error ", error, " ρ ", spec_rad)

        error_last = error
        aflux_last[:] = aflux[:]
        l += 1


    zeroth, first, second = compute_volume_moments(aflux)
    #Rem_l, Rem_r = compute_bound_rem(aflux, zeroth, first)

    #diff = build_diffusion_mat()

    #zeroth_new = np.linalg.solve(diff, diff_b_simple_palmer_bounds(second, Rem_l, Rem_r))
    #zeroth_new_dsa = np.linalg.solve(diff, diff_b_simple_revised(np.zeros_like(second)))

    #plt.figure()
    #plt.plot(np.linspace(0, N_cell, 2*N_cell), zeroth, label="OCI converged")
    #plt.plot(np.linspace(0, N_cell, 2*N_cell), zeroth_new, '.-', label="Palmer")
    #plt.plot(np.linspace(0, N_cell, 2*N_cell), zeroth_new_dsa, '--', label="Oiginal")
    #plt.title("ACCELERATION SOLUTIONS AT CONVERGENCE")
    #plt.xlabel("Distance")
    #plt.ylabel("φ")
    #plt.legend()
    #plt.show()

        
    #if (af_printer):
    #    plt.figure()
    #    for m in range(N_angle):
    #        af = np.zeros(2*N_cell)
    #        for i in range(N_cell):
    #            af_i = 2*i*N_angle + 2*m
    #            af[2*i] = aflux[af_i]
    #            af[2*i+1] = aflux[af_i+1]
    #        plt.plot(np.linspace(0, N_cell, 2*N_cell), af, label='m={}'.format(m))
    #    plt.xlabel("cell [j]")
    #    plt.ylabel("φ")
    #    plt.title("Final L={}\n S{}, σ={}, c={}, Δx={}".format(l, N_angle, sigma, sigma_s/sigma, dx))
    #    #plt.ylim([0,np.max(aflux)*1.5])
    #    plt.legend()
    #    plt.show()


    zeroth, first, second = compute_volume_moments(aflux)

    #print(zeroth[0], zeroth[-1])
    #print(first[0], first[-1])
    #print(second[0], second[-1])

    #plt.figure()
    #plt.plot(np.linspace(0, N_cell, 2*N_cell), zeroth)
    #plt.xlabel("cell [j]")
    #plt.ylabel("φ")
    #plt.title("Final L={}\n S{}, σ={}, c={}, Δx={}".format(l, N_angle, sigma, sigma_s/sigma, dx))
    #plt.ylim([0,np.max(zeroth)*1.5])
    #plt.show()

    return(zeroth, np.average(spec_rad_list), l)


td_exp = True
ss_exp = False

if ss_exp:

    source = 0.0

    dx = .1
    Len = 25
    N_cell = int(Len/dx)
    N = 2*N_angle*int(Len/dx)

    sigma_s = sigma*0.0
    sigma_a = sigma-sigma_s
    D = 1/(3*sigma)

    af_l = 0
    af_r = 0

    J_l = 0
    J_r = 0

    N_mfp = 10
    N_c = 15

    mfp_range = np.logspace(-1,1,N_mfp)
    c_range = np.linspace(0,1,N_c)

    spec_rad_pacc = np.zeros((N_mfp, N_c))
    spec_rad_oci = np.zeros((N_mfp, N_c))

    printer = False

    for k in range(N_mfp):
        for h in range(N_c):

            dx = mfp_range[k]/sigma
            N_cell = int(Len/dx)
            N = 2*N_angle*int(Len/dx)

            sigma_s = sigma*c_range[h]
            sigma_a = sigma-sigma_s
            D = 1/(3*sigma)

            SET_palmer_acc = False
            sf_oci, spec_rad_oci[k,h], oci_i = transport()

            SET_palmer_acc = True
            sf_palmer, spec_rad_pacc[k,h], sosa_i = transport()

            if spec_rad_pacc[k,h] > 1:
                spec_rad_pacc[k,h] = 1

            print("δ={}, c={}, OCI took {}({}), Palmer Acc took {}({})".format(mfp_range[k], c_range[h], oci_i, spec_rad_oci[k,h], sosa_i, spec_rad_pacc[k,h] ))
    
    np.savez("numerical_exp", mfp=mfp_range, c=c_range, spec_rad_oci=spec_rad_oci, spec_rad_pacc=spec_rad_pacc)
    
    print()
    print("Maximum ρ for SMM: %.3f, for OCI: %.3f"%(np.max(spec_rad_pacc), np.max(spec_rad_oci)))
    print()

    c, mfp = np.meshgrid(c_range, mfp_range)

    fig, (axs) = plt.subplots(ncols=2, layout="constrained")
    ax1 = axs[0]
    ax2 = axs[1]
    surf = ax1.contourf(mfp, c, spec_rad_pacc, levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)
    
    surf2 = ax2.contourf(mfp, c, spec_rad_oci, levels=100, vmin=0, vmax=1.0, cmap=cm.viridis, antialiased=True)

    fig.colorbar(surf2, ax=axs.ravel().tolist(), label=r'$\rho_e$')

    surf.set_edgecolor("face")
    surf2.set_edgecolor("face")

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    ax1.set_xlabel(r"$\delta$")
    ax1.set_ylabel(r"$c$")
    ax2.set_xlabel(r"$\delta$")

    ax1.set_title(r"2$^{nd}$ Moment Method")
    ax2.set_title("OCI")

    plt.gcf().set_size_inches(8, 4)
    ax2.label_outer()
    plt.savefig("smm_acc.pdf", ticks=[0, .2, .4, .6, .8, 1.0])
    plt.show()


    rho_acc = spec_rad_oci/spec_rad_pacc

    plt.figure()
    fig, (axs) = plt.subplots(layout="constrained")
    ax1 = axs
    surf = ax1.contourf(mfp, c, rho_acc, levels=100, cmap=cm.viridis, antialiased=True)
    ax1.contour(mfp, c, rho_acc, levels=[1.0], colors=('w',), linewidths=(3,))
    surf.set_edgecolor("face")
    ax1.set_xlabel(r"$\delta$")
    ax1.set_ylabel(r"$c$")
    ax1.set_xscale('log')
    fig.colorbar(surf, label=r'$\rho_{OCI}/\rho_{SMM}$')
    plt.savefig("rho_ratio.pdf")
    plt.show()

if td_exp:

    source = 0.0

    dx = .1
    N_cell = 50
    #Len = 1
    #N_cell = int(Len/dx)
    N = 2*N_angle*N_cell

    sigma_const = 1.0
    sigma_s = sigma*0.0
    sigma_a = sigma-sigma_s
    D = 1/(3*sigma)

    af_l = 0
    af_r = 0

    J_l = 0
    J_r = 0

    N_mfp = 10
    N_c = 6
    N_mft = 6

    mfp_range = np.logspace(-1,1,N_mfp)
    c_range = np.linspace(0,1,N_c)
    mft_range = np.logspace(-4,0,N_mft)

    spec_rad_pacc = np.zeros((N_mfp, N_c, N_mft))
    spec_rad_oci = np.zeros((N_mfp, N_c, N_mft))

    printer = False

    for t in range(N_mft):
        print("\n\n===============New MFT!===============")
        print(mft_range[t], sigma_const + 1/mft_range[t], " largest N cell ", int(Len/(mfp_range[0]/(sigma_const + 1/mft_range[t]))))
        print("\n\n")

        for k in range(N_mfp):
            for h in range(N_c):

                sigma = 1 + 1/mft_range[t]
                print(sigma*mft_range[t], mft_range[t], sigma)

                dx = mfp_range[k]/sigma
                Len = N_cell*dx
                #N_cell = int(Len/dx)
                #N = 2*N_angle*int(Len/dx)

                sigma_s = sigma*c_range[h]
                sigma_a = sigma-sigma_s
                D = 1/(3*sigma)

                SET_palmer_acc = False
                sf_oci, spec_rad_oci[k,h,t], oci_i = transport()

                SET_palmer_acc = True
                sf_palmer, spec_rad_pacc[k,h,t], sosa_i = transport()

                if spec_rad_pacc[k,h,t] > 1:
                    spec_rad_pacc[k,h,t] = 1

                print("τ=%.1e, δ=%.1e, c=%.1f, OCI took %4d (%.4f), Palmer Acc took %4d (%.4f)"%(mft_range[t], mfp_range[k], c_range[h], oci_i, spec_rad_oci[k,h,t], sosa_i, spec_rad_pacc[k,h,t] ))
    
    np.savez("td_exp", mfp=mfp_range, c=c_range, mft=mft_range, spec_rad_oci=spec_rad_oci, spec_rad_pacc=spec_rad_pacc)
    
    print()
    print("Maximum ρ for SMM: %.3f, for OCI: %.3f"%(np.max(spec_rad_pacc), np.max(spec_rad_oci)))
    print()