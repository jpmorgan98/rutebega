# Fixed-Point "GMRES" Accelerator for Transport / Source Iteration

This repository contains a small, reusable accelerator for fixed-point solvers such as source iteration or transport sweeps:

```python
from fixed_point_gmres import make_next_gmres

next_gmres = make_next_gmres(m=5, damping=1.0)

xo = initial_guess
while not converged:
    xn = transport_loop(xo)   # raw sweep / fixed-point update
    xn = next_gmres(xn, xo)   # accelerated iterate
    xo = xn
```

## What this code is, and what it is not

The module is named `fixed_point_gmres.py`, but mathematically it is **not classical matrix GMRES**. Classical GMRES assumes a linear system

\[
A x = b
\]

and builds an orthonormal Krylov basis from repeated applications of the linear operator \(A\), then chooses the iterate whose residual is minimal over that Krylov subspace [1].

This module is for the different situation where you **do not have \(A\)**, but you **do** have a fixed-point map

\[
x_{k+1} = G(x_k),
\]

or, operationally, a transport loop that takes an old iterate `xo` and returns a raw new iterate `xn`.

In that setting, the code uses a **small-history residual-minimization method in the Anderson / DIIS family** [2][3][4]. On linear problems, Anderson acceleration is closely connected to GMRES; in particular, Walker and Ni show that untruncated Anderson acceleration is, in a precise sense, essentially equivalent to GMRES on linear problems [3]. That is the reason the module is reasonably described as *GMRES-like* even though it does not explicitly build a Krylov basis or require an operator \(A\) [1][3].

## High-level idea

At iteration \(k\), your solver produces a raw next iterate

\[
x_k^{\mathrm{raw}} = G(x_{k-1}).
\]

If you do not provide an explicit residual, the code uses the natural fixed-point residual-like quantity

\[
r_k = x_k^{\mathrm{raw}} - x_{k-1}.
\]

If you *can* compute a better physics-based residual, you can pass it explicitly:

```python
xn = next_gmres(xn, xo, residual=my_residual)
```

The accelerator stores the most recent \(m\) pairs

\[
(x_i, r_i), \qquad i = k-m+1, \dots, k,
\]

and computes coefficients \(c_i\) that minimize the norm of the combined residual subject to the coefficients summing to one:

\[
\min_{c_1,\dots,c_m} \left\| \sum_{i=1}^{m} c_i r_i \right\|_2
\qquad \text{subject to} \qquad
\sum_{i=1}^{m} c_i = 1.
\]

This is the standard DIIS / Anderson residual-minimization viewpoint [2][3][4].

Once the coefficients are found, the code forms the accelerated iterate as the same linear combination of the stored iterates:

\[
x^{\mathrm{acc}} = \sum_{i=1}^{m} c_i x_i.
\]

Finally, it applies optional damping:

\[
x^{\mathrm{out}} = (1-\omega) x^{\mathrm{raw}} + \omega x^{\mathrm{acc}},
\]

where `damping = \omega`.

## The algebra used in the code

Define the residual Gram matrix

\[
G_{ij} = r_i^T r_j.
\]

Then the constrained minimization problem above can be written using a Lagrange multiplier \(\lambda\):

\[
\mathcal{L}(c,\lambda) = c^T G c - \lambda (\mathbf{1}^T c - 1).
\]

Taking derivatives gives the saddle-point system

\[
\begin{bmatrix}
G & \mathbf{1} \\
\mathbf{1}^T & 0
\end{bmatrix}
\begin{bmatrix}
c \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
0 \\
1
\end{bmatrix}.
\]

That is exactly the small dense system assembled in the implementation, except that the code optionally adds a tiny diagonal regularization to the Gram matrix,

\[
G \leftarrow G + \epsilon I,
\]

to reduce sensitivity when the residual history becomes nearly linearly dependent.

After solving for \(c\), the code computes

\[
x^{\mathrm{acc}} = X c,
\]

where \(X\) is the matrix whose columns are the stored iterate vectors.

## Why this is useful for transport-style fixed-point solves

In many transport or source-iteration codes, the expensive part is the sweep / transport solve itself, while forming or storing a global Jacobian or transport operator may be impractical. Anderson / DIIS methods are designed exactly for this setting: they accelerate fixed-point iterations using only the recent iterates and residuals, without requiring an explicit Jacobian or matrix operator [2][3][5].

This makes the method attractive when you want a drop-in accelerator with a calling pattern like

```python
xn = transport_loop(xo)
xn = next_gmres(xn, xo)
```

rather than a classical linear solver interface of the form `gmres(A, b)`.

## API

### `make_next_gmres(...)`

Factory function returning a stateful callable.

```python
next_gmres = make_next_gmres(
    m=5,
    damping=1.0,
    regularization=1.0e-12,
    max_weight_norm=1.0e6,
)
```

Parameters:

- `m`: maximum history length.
- `damping`: blend between raw and accelerated iterate.
- `regularization`: diagonal shift added to the residual Gram matrix.
- `max_weight_norm`: safety guard; if the DIIS weights get too large, the code falls back to the raw iterate.

### `next_gmres(xn, xo, residual=None)`

Returns an accelerated iterate.

- `xn`: raw next iterate from the transport loop.
- `xo`: previous iterate.
- `residual`: optional user-supplied residual; if omitted, the method uses `xn - xo`.

### `reset()`

Clears the stored history. Call this before reusing the same accelerator object for a different solve or a different problem size.

## Mapping the code to the math

The implementation stores:

- `self._x_hist`: recent iterate vectors \(x_i\)
- `self._r_hist`: recent residual vectors \(r_i\)

Then it builds:

- `R = [r_1 \; r_2 \; \dots \; r_m]`
- `G = R^T R`

and solves the augmented system

\[
\begin{bmatrix}
G & \mathbf{1} \\
\mathbf{1}^T & 0
\end{bmatrix}
\begin{bmatrix}
c \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
0 \\
1
\end{bmatrix}
\]

for the coefficients \(c\). The accelerated iterate is then built as

\[
x^{\mathrm{acc}} = X c,
\]

and the returned iterate is

\[
x^{\mathrm{out}} = (1-\omega)x^{\mathrm{raw}} + \omega x^{\mathrm{acc}}.
\]

## Practical recommendations

### 1. Start with a short history

A history length of `m=3` to `m=6` is a good starting range. Larger histories can help, but they also make the small dense system more ill-conditioned [3].

### 2. Prefer a physically meaningful residual if available

The default `xn - xo` works because it is the naturally available fixed-point residual-like quantity. But if your transport solver has a more meaningful residual, that is usually the better quantity to minimize.

### 3. Use damping if the iteration becomes noisy

If acceleration overshoots or becomes erratic, reduce the damping factor, for example to `0.5` or `0.8`.

### 4. Reset between unrelated solves

The object is stateful. Do not carry history from one unrelated solve to the next.

## Minimal example

```python
import numpy as np
from fixed_point_gmres import make_next_gmres


def transport_loop(xo):
    # Replace this with your actual transport / source iteration sweep
    return 0.8 * xo + 0.2


xo = 100.0 * np.random.random(100)
next_gmres = make_next_gmres(m=5, damping=1.0)

tol = 1.0e-10
max_its = 200

for it in range(max_its):
    xn = transport_loop(xo)
    xn = next_gmres(xn, xo)

    res = np.linalg.norm(transport_loop(xn) - xn)
    if res < tol:
        xo = xn
        break

    xo = xn
```

## References

[1] Y. Saad and M. H. Schultz, *GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems*, SIAM Journal on Scientific and Statistical Computing, 7(3), 856-869, 1986. DOI: `10.1137/0907058`.

[2] D. G. Anderson, *Iterative Procedures for Nonlinear Integral Equations*, Journal of the ACM, 12(4), 547-560, 1965. DOI: `10.1145/321296.321305`.

[3] H. F. Walker and P. Ni, *Anderson Acceleration for Fixed-Point Iterations*, SIAM Journal on Numerical Analysis, 49(4), 1715-1735, 2011. DOI: `10.1137/10078356X`.

[4] P. Pulay, *Convergence Acceleration of Iterative Sequences. The Case of SCF Iteration*, Chemical Physics Letters, 73(2), 393-398, 1980. DOI: `10.1016/0009-2614(80)80396-4`.

[5] H.-r. Fang and Y. Saad, *Two Classes of Multisecant Methods for Nonlinear Acceleration*, Numerical Linear Algebra with Applications, 16(3), 197-221, 2009. DOI: `10.1002/nla.617`.
