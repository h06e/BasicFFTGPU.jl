# GPU benchmark on a 512^3 grid: spherical inclusion in a matrix.
#
#   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'   (first time)
#   julia --project=benchmark benchmark/benchmark_512.jl [n]
#
# Float32, Barzilai-Borwein acceleration, tol = 1e-6, both Green operators,
# macroscopic strain and stress loading. Timings exclude compilation (warm-up
# on a small grid) and microstructure generation.

using BasicFFTGPU
using CUDA
using Printf

n = isempty(ARGS) ? 512 : parse(Int, ARGS[1])
tol = 1e-6

"Spherical inclusion (label 1, volume fraction ~ 11%) centred in an n^3 matrix (label 0)."
function sphere_phases(n)
    c = (n + 1) / 2
    r2 = (0.3n)^2
    return [Int32((i - c)^2 + (j - c)^2 + (k - c)^2 < r2) for i in 1:n, j in 1:n, k in 1:n]
end

materials = Dict(0 => Isotropic(E=1.0, nu=0.3),      # matrix
                 1 => Isotropic(E=100.0, nu=0.2))    # inclusion, contrast 100
loadings = ["strain E11" => StrainLoading([0.01, 0, 0, 0, 0, 0]),
            "stress S11" => StressLoading([1.0, 0, 0, 0, 0, 0])]

# warm-up (compilation)
small = PhaseMaterials(sphere_phases(16), materials)
for g in (:continuous, :staggered), (_, load) in loadings
    solve(small, load; green=g, device=CuArray, keep_fields=false)
end

mat = PhaseMaterials(sphere_phases(n), materials)
println("GPU: ", CUDA.name(CUDA.device()), " | grid $n^3 | Float32 | tol = $tol")
@printf("%-12s %-12s %6s %10s %12s\n", "green", "loading", "iter", "time [s]", "ms / iter")
for g in (:continuous, :staggered)
    ws = Workspace(mat; green=g, device=CuArray)
    for (name, load) in loadings
        CUDA.synchronize()
        t = @elapsed begin
            r = solve!(ws, load; tol=tol, keep_fields=false)
            CUDA.synchronize()
        end
        r.converged || @warn "not converged" g name
        @printf("%-12s %-12s %6d %10.2f %12.1f\n", g, name, r.iterations, t, 1000t / max(r.iterations, 1))
    end
    used = CUDA.total_memory() - CUDA.free_memory()
    @printf("  device memory in use with workspace: %.1f GiB\n", used / 2^30)
    ws = nothing
    GC.gc(); CUDA.reclaim()
end
