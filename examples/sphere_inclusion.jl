# Spherical inclusion in a matrix: build a .vti microstructure, load it,
# solve under strain and stress loading, export the fields to ParaView.
#
#   julia --project=. examples/sphere_inclusion.jl          (CPU)
#   julia --project=. examples/sphere_inclusion.jl gpu      (GPU, needs CUDA in the environment)

using BasicFFTGPU

use_gpu = "gpu" in ARGS
if use_gpu
    using CUDA
    device = CuArray
else
    device = Array
end

#! 1. Write a microstructure file (normally you'd already have one)
n = 64
phases = [(i - 32.5)^2 + (j - 32.5)^2 + (k - 32.5)^2 < 20^2 ? 1 : 0 for i in 1:n, j in 1:n, k in 1:n]
grid = VoxelGrid((n, n, n); spacing=(1 / n, 1 / n, 1 / n))
write_vtk("sphere_micro", Dict("phase" => Int32.(phases)), grid)

#! 2a. Material list: one material per phase label
materials = Dict(0 => Isotropic(E=1.0, nu=0.3),       # matrix
                 1 => Isotropic(E=100.0, nu=0.2))     # inclusion
mat = load_phase_materials("sphere_micro.vti", materials)

#! 2b. Or voxel-wise constants read from fields of a .vti file
E = [p == 1 ? 100.0 : 1.0 for p in phases]
nu = [p == 1 ? 0.2 : 0.3 for p in phases]
write_vtk("sphere_constants", Dict("E" => E, "nu" => nu), grid)
mat_voxel = load_voxel_materials("sphere_constants.vti"; E="E", nu="nu")

#! 3. Macroscopic strain loading (Kelvin 6-vector), Moulinec-Suquet Green operator (default)
r = solve(mat, StrainLoading([0.01, 0, 0, 0, 0, 0]); device=device, verbose=true)
println("mean stress: ", r.mean_stress)
write_vtk("sphere_strain_loading", r; material=mat)

#! 4. Macroscopic stress loading, staggered-grid Green operator
r = solve(mat_voxel, StressLoading([1.0, 0, 0, 0, 0, 0]); green=:staggered, device=device)
println("mean strain under uniaxial stress: ", r.mean_strain, " (", r.iterations, " iterations)")
write_vtk("sphere_stress_loading", r; material=mat_voxel)

#! 5. Effective stiffness (Kelvin notation)
C = effective_stiffness(mat; device=device)
display(round.(C, sigdigits=4))
