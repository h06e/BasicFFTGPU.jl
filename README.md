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
- voxel-wise J2 elastoplasticity with Voce isotropic hardening, incremental loading

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

## Elastoplasticity

`VoxelPlasticMaterials`: small-strain J2 plasticity, isotropic hardening
`R(p) = sigma0 + H p + Q (1 - exp(-b p))`, radial return per voxel. Constants are 3D arrays
or scalars. Solve load increments on one workspace, warm-started from the previous
increment, and commit the internal state after each one:

```julia
mat = VoxelPlasticMaterials(E=E_field, nu=0.3, sigma0=s0_field, Q=200.0, b=20.0, H=0.0)
ws = Workspace(mat; device=CuArray)
for e in range(0, 0.01; length=21)[2:end]
    r = solve!(ws, StrainLoading([e, 0, 0, 0, 0, 0]); warm_start=true)
    commit!(ws)                                   # plastic state of this increment -> reference
end
p = cumulated_plastic_strain(ws); εp = plastic_strain(ws)
write_vtk("step", r; material=mat, workspace=ws)  # + plastic_strain, cumulated_plastic_strain
```

## Benchmark

Spherical inclusion (volume fraction ≈ 11 %, Young's modulus contrast 100) in a **512³** grid,
Float32, Barzilai-Borwein acceleration, `tol = 1e-6`. Times exclude compilation and
microstructure generation.

- GPU: NVIDIA RTX 6000 Ada (48 GB)
- CPU: AMD Ryzen Threadripper PRO 7985WX, 64 cores (64 Julia threads, 64 FFTW threads)

| Green operator | Loading | Iterations GPU | Time GPU [s] | Iterations CPU | Time CPU [s] | Speed-up GPU / CPU |
|---|---|---:|---:|---:|---:|---:|
| Moulinec-Suquet (`:continuous`) | strain E11 | 54 | 6.9 | 61 | 158.2 | 22.9× |
| Moulinec-Suquet (`:continuous`) | stress S11 | 79 | 10.7 | 81 | 234.7 | 22.0× |
| staggered grid (`:staggered`) | strain E11 | 51 | 6.8 | 49 | 132.6 | 19.4× |
| staggered grid (`:staggered`) | stress S11 | 103 | 14.5 | 87 | 262.5 | 18.1× |

Per iteration: 128–141 ms on GPU, 2.6–3.0 s on CPU. GPU memory used: 14.1 GiB
(≈ 113 bytes per voxel: 18 real and 6 half-size complex fields, plus cuFFT plans).

GPU and CPU macroscopic responses (mean strain and stress) agree within 1.5e-6 relative.
Iteration counts differ slightly because Float32 round-off differs between cuFFT and FFTW,
and the Barzilai-Borwein step amplifies these small differences along the iterations.

Reproduce with:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia -t auto --project=benchmark benchmark/benchmark_512.jl            # n = 512, gpu and cpu
julia -t auto --project=benchmark benchmark/benchmark_512.jl 256 gpu    # other size / device
```

## Tests

```julia
pkg> test BasicFFTGPU
```

Checks include exactness on laminates (both Green operators), consistency between strain,
stress and mixed loadings, phase-list vs voxel-wise materials, and VTK round trips.
