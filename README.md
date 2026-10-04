# BasicFFTGPU

Basic FFT-based homogenization solver (Moulinec & Suquet) for linear elastic voxel
microstructures, on CPU (FFTW) or NVIDIA GPU (cuFFT).

- microstructure import / result export in VTK ImageData (`.vti`)
- Green operator: continuous Moulinec-Suquet (default) or staggered grid
- Kelvin notation for every tensor (loadings, results, stiffness matrices)
- macroscopic strain, stress or mixed loading
- materials from a phase map + material list (isotropic or anisotropic), or voxel-wise
  isotropic constants read from a `.vti` file
- basic scheme with optional Barzilai-Borwein acceleration (default on)

## Install

```julia
pkg> dev ~/Documents/basicFFTGPU
pkg> add CUDA        # only for GPU runs
```

## Usage

```julia
using BasicFFTGPU
using CUDA                                          # optional, for device=CuArray

# phase map + material list
mat = load_phase_materials("micro.vti", Dict(0 => Isotropic(E=1.0, nu=0.3),
                                             1 => Isotropic(kappa=80.0, mu=40.0)))
# or voxel-wise constants (field names in the file)
mat = load_voxel_materials("constants.vti"; E="E", nu="nu")

r = solve(mat, StrainLoading([0.01, 0, 0, 0, 0, 0]);  # Kelvin 6-vector or 3x3 matrix; or StressLoading, MixedLoading
          green = :continuous,                        # Moulinec-Suquet (default) or :staggered
          device = CuArray,                           # Array (default) for CPU
          precision = Float32, tol = 1e-6, maxiter = 1000, verbose = true)

r.mean_stress, r.mean_strain, r.iterations
write_vtk("result", r; material=mat)                  # -> result.vti, open in ParaView

C = effective_stiffness(mat; device=CuArray)          # 6x6, Kelvin notation
```

All tensors are in Kelvin notation: a symmetric tensor is `(a11, a22, a33, √2 a23, √2 a13, √2 a12)`
(`kelvin(a)`, `kelvin_to_tensor(v)`), a stiffness is the 6x6 matrix with `σ = C * ε`.
Anisotropic phases: `Anisotropic(C)`, with helpers `cubic_stiffness(c11, c12, c44)` and
`rotate_stiffness(C, R)`.
In-memory arrays work too: `PhaseMaterials(phases, materials; grid)`,
`VoxelMaterials(; E, nu, grid)`. See [docs/CONVENTIONS.md](docs/CONVENTIONS.md) and
[examples/sphere_inclusion.jl](examples/sphere_inclusion.jl) and
[examples/anisotropic_polycrystal.jl](examples/anisotropic_polycrystal.jl) (Voronoi polycrystal of
randomly oriented cubic grains).

## Benchmark

Spherical inclusion (volume fraction ≈ 11 %, Young's modulus contrast 100) in a **512³** grid,
NVIDIA RTX 6000 Ada (48 GB), Float32, Barzilai-Borwein acceleration, `tol = 1e-6`.
Times exclude compilation and microstructure generation.

| Green operator | Loading | Iterations | Time [s] | ms / iteration |
|---|---|---:|---:|---:|
| Moulinec-Suquet (`:continuous`) | strain E11 | 54 | 6.9 | 128 |
| Moulinec-Suquet (`:continuous`) | stress S11 | 86 | 11.6 | 135 |
| staggered grid (`:staggered`) | strain E11 | 51 | 6.8 | 134 |
| staggered grid (`:staggered`) | stress S11 | 104 | 14.7 | 141 |

GPU memory used: 14.1 GiB (≈ 27 bytes per voxel). Reproduce with:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/benchmark_512.jl        # optional grid size argument
```

## Tests

```julia
pkg> test BasicFFTGPU
```

Checks include exactness on laminates (both Green operators), consistency between strain,
stress and mixed loadings, phase-list vs voxel-wise materials, and VTK round trips.
