# GPU vs CPU benchmark on a 512^3 grid: spherical inclusion in a matrix.
#
#   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'       (first time)
#   julia -t auto --project=benchmark benchmark/benchmark_512.jl [n] [devices]
#
# n defaults to 512, devices to "gpu,cpu". The CPU runs use all Julia threads
# (KernelAbstractions kernels) and as many FFTW threads.
#
# Float32, Barzilai-Borwein acceleration, tol = 1e-6, both Green operators,
# macroscopic strain and stress loading. Timings exclude compilation (warm-up
# on a small grid) and microstructure generation.

using BasicFFTGPU
using CUDA
using FFTW
using LinearAlgebra
using Printf

n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
devices = length(ARGS) >= 2 ? split(ARGS[2], ",") : ["gpu", "cpu"]
tol = 1e-6
FFTW.set_num_threads(Threads.nthreads())

device_array(d) = d == "gpu" ? CuArray : Array
sync(d) = d == "gpu" ? CUDA.synchronize() : nothing
device_name(d) = d == "gpu" ? CUDA.name(CUDA.device()) :
                 "$(Sys.cpu_info()[1].model), $(Threads.nthreads()) threads"

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
greens = (:continuous, :staggered)

# warm-up (compilation)
small = PhaseMaterials(sphere_phases(16), materials)
for d in devices, g in greens, (_, load) in loadings
    solve(small, load; green=g, device=device_array(d), keep_fields=false)
end

mat = PhaseMaterials(sphere_phases(n), materials)
results = Dict()   # (green, loading, device) => (iterations, time, mean strain, mean stress)

println("grid $n^3 | Float32 | tol = $tol")
for d in devices
    println(uppercase(d), ": ", device_name(d))
end
println()
@printf("%-6s %-12s %-12s %6s %10s %12s\n", "device", "green", "loading", "iter", "time [s]", "ms / iter")
for d in devices, g in greens
    ws = Workspace(mat; green=g, device=device_array(d))
    for (name, load) in loadings
        sync(d)
        t = @elapsed begin
            r = solve!(ws, load; tol=tol, keep_fields=false)
            sync(d)
        end
        r.converged || @warn "not converged" d g name
        results[(g, name, d)] = (r.iterations, t, r.mean_strain, r.mean_stress)
        @printf("%-6s %-12s %-12s %6d %10.2f %12.1f\n", d, g, name, r.iterations, t, 1000t / max(r.iterations, 1))
    end
    if d == "gpu"
        @printf("       device memory in use with workspace: %.1f GiB\n",
                (CUDA.total_memory() - CUDA.free_memory()) / 2^30)
    end
    ws = nothing
    GC.gc()
    d == "gpu" && CUDA.reclaim()
end

if issubset(["gpu", "cpu"], devices)
    println("\nGPU vs CPU")
    @printf("%-12s %-12s %10s %10s %12s %16s\n", "green", "loading", "iter cpu", "iter gpu",
            "speed-up", "rel. diff mean")
    for g in greens, (name, _) in loadings
        ic, tc, Ec, Sc = results[(g, name, "cpu")]
        ig, tg, Eg, Sg = results[(g, name, "gpu")]
        # relative difference of the macroscopic response (strain and stress)
        diff = max(norm(Ec - Eg) / norm(Ec), norm(Sc - Sg) / norm(Sc))
        @printf("%-12s %-12s %10d %10d %11.1fx %16.1e\n", g, name, ic, ig, tc / tg, diff)
    end
end
