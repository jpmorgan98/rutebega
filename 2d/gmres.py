import numpy as np

# initializing a storage array which will be appended
storage = np.array([])

# target is a vector of zeros

def iteration(x1, x2, restart=5):

    res = x1-x2

    target = np.zeros_like(x1)

    storage = np.append(res, axis=1)

    z, residuals, rank, s = np.linalg.lstsq(storage, b, rcond=None)

    return(x1+z)


if __name__ == '__main__':

    max_its = 1000

    xo = 100*np.random.random(100)

    for i in range(max_its):
        xn = 1 - xo
        xn = iteration(xn, xo)
        
        if xn < 1:
            print("CONVERGED! in {} iterations".format(i))
            break
        
        xn = xo



    print("converged in ")
