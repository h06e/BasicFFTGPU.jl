# Conventions

## Kelvin notation

Every tensor, user side and device side, is in Kelvin notation (no Voigt anywhere):

- symmetric second-order tensor (strain, stress): the 6-vector
  `(a11, a22, a33, √2 a23, √2 a13, √2 a12)` — `kelvin(a)` / `kelvin_to_tensor(v)`;
- stiffness: the 6x6 matrix `C` with `σ = C * ε` on Kelvin vectors
  (`isotropic_stiffness`, `cubic_stiffness`, `Anisotropic(C)`, `effective_stiffness`).

The basis is orthonormal, so `a : b = dot(a, b)` (used in the residual norms and the
Barzilai-Borwein step), stiffness eigenvalues are the Kelvin moduli, and a rotation `R`
acts through the orthogonal 6x6 matrix `rotation_kelvin(R)` (`rotate_stiffness`).
Loadings, `mean_strain`, `mean_stress` and the `strain`/`stress` fields are Kelvin vectors.

The only non-Kelvin outputs are in the `.vti` export: the scalar fields `strain_ij`,
`stress_ij` (and `plastic_strain_ij`), one per tensor component, and the `strain`/`stress`
arrays, which hold plain tensor components in ParaView's symmetric-tensor order
(XX YY ZZ XY YZ XZ) so that ParaView computes eigenvalues correctly;
`strain_kelvin`/`stress_kelvin` are written too.

## Grid and frequencies

Voxels are the cells of the image; a field is an `nx x ny x nz` array, `nz = 1` for 2D
(plane strain). Spacing `h` comes from the VTK file and enters the frequencies:

| scheme        | frequency along direction d                  | Nyquist (even n)  |
|---------------|----------------------------------------------|-------------------|
| `:continuous` (default) | `k / (n_d h_d)`                    | zeroed            |
| `:staggered`  | `(exp(-2 i pi k / n_d) - 1) / h_d`           | kept (well defined) |

The Green operator kernels are those of Kairotop / benchmark_fft, generalized to
non-cubic grids and non-unit spacing. The zero frequency is always zeroed; the mean
strain is handled by the solver.

## Loading and iteration

`eps <- eps - rho * r`, with `r = Gamma0 sigma` for the fluctuation and, on
stress-controlled components `s`, the mean part `-(C0_ss)^-1 (S_s - <sigma>_s)`.
`rho` is the Barzilai-Borwein step (`accelerate=true`) or 1 (basic scheme).

Convergence: `sqrt(<|Gamma0 sigma|^2>) / |<eps>| < tol` and, if some components are
stress controlled, `|<sigma>_s - S_s| / |<sigma>| < tol`. For plastic materials the mean strain can
vanish (unloading), so `|<eps>|` is replaced by the rms strain `sqrt(<eps:eps>)`.

## Plasticity (`VoxelPlasticMaterials`)

Committed state `(eps_p_n, p_n)`, trial state `(eps_p, p)` recomputed at every stress
evaluation by radial return from the committed one; `commit!(ws)` copies trial to
committed. Return mapping: Newton on `q_trial - 3 mu dp = R(p_n + dp)`, monotone since
`R` is concave. The reference medium uses the elastic constants.

## Reference medium

Isotropic `(kappa0, mu0)`: midpoint of the extreme bulk and shear moduli. With anisotropic
phases, midpoint of the extreme Kelvin moduli (eigenvalues of the Kelvin stiffness) for both
`3 kappa0` and `2 mu0`. Override with `reference=(kappa0, mu0)`.

## VTK

XML ImageData (`.vti`), voxel values as **cell data** (WriteVTK.jl / ReadVTK.jl).
Files with point data only are accepted, each point being one voxel. Export: see
"Kelvin notation" above.
