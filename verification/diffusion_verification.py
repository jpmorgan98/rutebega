import numpy as np
import matplotlib.pyplot as plt


# Problem definitions and global variable initialization
dx = 0.1

Len = 4
N_cell = int(Len/dx)
N = 2*N_cell

sigma = 10
sigma_s = .5*sigma
sigma_a = sigma - sigma_s
D = 1/(3*sigma)

source = 20*np.ones(N_cell)

J_l = 0
J_r = 0

bcl = 'ref'
bcr = 'ref'

# quadrature limits
delta = 1/2
gamma = 1/4

x = np.linspace(0,Len,N_cell)

def linear_mms(source):

    manufactured_solution = np.zeros_like(source)

    for i in range(N_cell):
        manufactured_solution[i] = .5*x[i] + 1
        source[i] = sigma_a*(x[i]/2+1)

    return(manufactured_solution)

def quadratic_mms(source):

    manufactured_solution = np.zeros_like(source)

    for i in range(N_cell):
        manufactured_solution[i] = -2*x[i]**2 + 1
        source[i] = -D*(-4) + sigma_a*(-2*x[i]**2 + 1)

    return(manufactured_solution)


def build_diffusion_mat():
    # building the diffusion A mat
    # this system is *very* sparse, doing this as dense is silly but ok for now

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
                A[2*i,2*i] = D/dx + sigma_a * dx/2
                A[2*i,2*i+1] = -D/dx

        else: #interior cell
            #cell j-1 info
            A[2*i,2*(i-1)]   = -delta*D/dx
            A[2*i,2*(i-1)+1] = delta*D/dx - gamma
            # cell j+1 info=
            A[2*i+1,2*(i+1)]   = -gamma+delta*D/dx
            A[2*i+1,2*(i+1)+1] = -delta*D/dx

    return(A)

def diffusion_solve():
    # building the diffusion matrix and solving the equation

    sf = np.random.random(N)
    second = np.zeros(N)

    b_vec = np.zeros(2*N_cell)
    for i in range(N_cell):
        if i == 0: # left hand bound, using bcs
            b_l = dx/2*source[i] + J_l #+ D/dx*(second[2*i+1]-second[2*i])
            b_r = dx/2*source[i] #- (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) + (delta*4*D)/dx * (second[2*(i+1)+1]-second[2*(i+1)])
            if bcl == 'ref':
                b_l = dx/2*source[i]

        elif i==N_cell-1: #right hand bcs
            b_l = dx/2*source[i] #+ (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (delta*4*D)/dx * (second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx/2*source[i] + J_r #- D/dx*(second[2*i+1]-second[2*i]) 

            if bcr == 'ref':
                b_r = dx/2*source[i]

        else: # interior cell
            b_l = dx/2*source[i] #+ (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (delta*4*D)/dx * (second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx/2*source[i] #- (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) + (delta*4*D)/dx * (second[2*(i+1)+1]-second[2*(i+1)])

        b_vec[2*i]   = b_l
        b_vec[2*i+1] = b_r

    diff = build_diffusion_mat()

    sf = np.linalg.solve(diff, b_vec)

    J = np.zeros_like(sf)
    for i in range(N_cell):
        J[2*i]   = -1/(3*dx*sigma)*(sf[2*i+1]-sf[2*i]) #- 4/(3*dx)*(second[2*i+1]-second[2*i])
        J[2*i+1] = -1/(3*dx*sigma)*(sf[2*i+1]-sf[2*i]) #- 4/(3*dx)*(second[2*i+1]-second[2*i])

    return(sf, J)


#
# Inf pure absorber via Boundary Conditions
#

# boundary conditions
af_l = source[0]/(2*sigma_a)
af_r = source[-1]/(2*sigma_a)

J_l = 0.5*af_l
J_r = 0.5*af_r

bcl = 'inc'
bcr = 'inc'

sf, J = diffusion_solve()

plt.figure()
plt.plot(np.linspace(0,N_cell, 2*N_cell), sf)
plt.ylabel("φ")
plt.xlabel("Cell")
plt.title("Inf Pure Absorber via Diffusion approximation \n (Incident BCs) Σ={}, c={}, Δx={}".format(sigma, sigma_s/sigma, dx))
plt.ylim([0, 1.25*np.max(sf)])
plt.show()


#
# Inf pure absorber via Reflecting BCs
#

bcl = 'ref'
bcr = 'ref'

sf_new, J_new = diffusion_solve()

plt.figure()
plt.plot(np.linspace(0,N_cell, 2*N_cell), sf)
plt.ylabel("φ")
plt.xlabel("Cell")
plt.title("Inf Pure Absorber via Diffusion approximation \n (Reflecting BCs) Σ={}, c={}, Δx={}".format(sigma, sigma_s/sigma, dx))
plt.ylim([0, 1.25*np.max(sf)])
plt.show()


#
# Linear MMS
#

man_lin_sol = linear_mms(source)

#source /= 2

bcl = 'inc'
bcr = 'inc'

# boundary conditions
af_l = source[0]/(2*sigma_a)
af_r = source[-1]/(2*sigma_a)

J_l = 1/4 - D/4
J_r = (Len/2+1)/4 + D/4

sf, J = diffusion_solve()

plt.figure()
plt.plot(np.linspace(0,Len, N_cell), man_lin_sol, '*', label=r"$\phi_{man}=\frac{x}{2}+1$")
plt.plot(np.linspace(0,Len, 2*N_cell), sf, '--', label='diffusion sol')
#plt.plot(np.linspace(0,Len, N_cell), source, label=r"$Q_{man}=\Sigma_a\left(\frac{x}{2}+1\right)$")
plt.ylabel("φ")
plt.xlabel("Distance [cm]")
plt.title("Linear MMS Verification \n (Prescribed Incident BCs) Σ={}, c={}, Δx={}".format(sigma, sigma_s/sigma, dx))
plt.legend()
plt.savefig("linear_mms.pdf")
#plt.show()

#
# Quadratic MMS
#

man_lin_sol = quadratic_mms(source)

bcl = 'inc'
bcr = 'inc'

J_l = 1/4
J_r = (-2*Len**2+1)/4 - 2*D*Len

sf, J = diffusion_solve()

plt.figure()

plt.plot(np.linspace(0,Len, N_cell), man_lin_sol, '*', label=r"$\phi_{man}=-2x^2+1$")
plt.plot(np.linspace(0,Len, 2*N_cell), sf, '--', label='diffusion sol')
#plt.plot(np.linspace(0,Len, N_cell), source, label=r"$Q_{man}=4D+\Sigma_a\left(-2x^2+1\right)$")
plt.ylabel("φ")
plt.xlabel("Distance [cm]")
plt.title("Quadratic MMS Verification \n (Prescribed Incident BCs) Σ={}, c={}, Δx={}".format(sigma, sigma_s/sigma, dx))
plt.legend()
#plt.ylim([0, 1.25*np.max(sf)])
plt.savefig("quadratic_mms.pdf")
#plt.show()


N_dx = 5
dxs = np.array([1, .5, .1, .05, .01, .005, .001])#np.logspace(-1,-4,N_dx)
error = np.zeros_like(dxs)

for i in range(N_dx):
    print("now running dx {}".format(dxs[i]))
    dx = dxs[i]
    N_cell = int(Len/dx)
    N = 2*N_cell
    source = 20*np.ones(N_cell)
    x = np.linspace(0,Len,N_cell)

    man_lin_sol = quadratic_mms(source)
    sf_comp = np.zeros_like(man_lin_sol)

    sf, J = diffusion_solve()

    for j in range(man_lin_sol.size):
        sf_comp[j] = (sf[2*j] + sf[2*j+1])/2

    error[i] = man_lin_sol[man_lin_sol.size // 2]-sf[sf.size // 2]
    
print(error)


plt.figure()
plt.plot(dxs, error)
plt.xscale('log')
plt.yscale("log")
plt.grid()
plt.show()