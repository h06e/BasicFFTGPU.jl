# Polycrystal of cubic (anisotropic) grains with random orientations.
#
# The microstructure is a periodic Voronoi tessellation stored as a grain-id
# map in a .vti file; each grain is one phase with its own rotated cubic
# stiffness (material list of `Anisotropic` constituents, Kelvin notation).
#
#   julia --project=. examples/anisotropic_polycrystal.jl          (CPU)
#   julia --project=. examples/anisotropic_polycrystal.jl gpu      (GPU, needs CUDA in the environment)

using BasicFFTGPU
using LinearAlgebra
using Random

use_gpu = "gpu" in ARGS
if use_gpu
    using CUDA
    device = CuArray
else
    device = Array
end

Random.seed!(1)

#! 1. Periodic Voronoi microstructure -> grain_ids.vti
n = 64
ngrains = 20
seeds = rand(3, ngrains)

"Index of the closest seed to point x, with periodic distances on the unit cube."
function closest_seed(x, seeds)
    d = [sum(abs2, (x .- seeds[:, g]) .- round.(x .- seeds[:, g])) for g in axes(seeds, 2)]
    return argmin(d)
end

grains = [closest_seed(([i, j, k] .- 0.5) ./ n, seeds) for i in 1:n, j in 1:n, k in 1:n]
grid = VoxelGrid((n, n, n); spacing=(1 / n, 1 / n, 1 / n))
write_vtk("grain_ids", Dict("grain" => Int32.(grains)), grid)

#! 2. Constituent: cubic copper single crystal (GPa), c11 = C1111, c12 = C1122, c44 = C2323
C_crystal = cubic_stiffness(168.4, 121.4, 75.4)

"Uniformly distributed random rotation (from a random unit quaternion)."
function random_rotation()
    q = normalize(randn(4))
    a, b, c, d = q
    return [a^2+b^2-c^2-d^2  2(b*c-a*d)       2(b*d+a*c);
            2(b*c+a*d)       a^2-b^2+c^2-d^2  2(c*d-a*b);
            2(b*d-a*c)       2(c*d+a*b)       a^2-b^2-c^2+d^2]
end

materials = Dict(g => Anisotropic(rotate_stiffness(C_crystal, random_rotation())) for g in 1:ngrains)
mat = load_phase_materials("grain_ids.vti", materials)

#! 3. Effective stiffness (Kelvin notation), Moulinec-Suquet Green operator (default)
C = effective_stiffness(mat; device=device, tol=1e-6)
println("Effective stiffness (Kelvin, GPa):")
display(round.(C, digits=2))

# Isotropic part of C: 3k = J::C, 2m = K::C / 5 (J, K: isotropic projectors in Kelvin)
J = zeros(6, 6); J[1:3, 1:3] .= 1 / 3
K = I(6) - J
k_eff = tr(J * C) / 3
m_eff = tr(K * C) / 10
println("isotropic part: kappa = $(round(k_eff, digits=2)) GPa, mu = $(round(m_eff, digits=2)) GPa")
println("anisotropy |C - C_iso| / |C| = ",
        round(norm(C - isotropic_stiffness(k_eff, m_eff)) / norm(C), sigdigits=3))

#! 4. Uniaxial tension under macroscopic stress control: S = 100 MPa e1 (x) e1
r = solve(mat, StressLoading([0.1, 0, 0, 0, 0, 0]); device=device, verbose=false)
println("mean strain (Kelvin) under 100 MPa uniaxial stress: ", round.(r.mean_strain, sigdigits=4),
        " (", r.iterations, " iterations)")
write_vtk("polycrystal_uniaxial", r; material=mat)   # open in ParaView, color by von_mises_stress
