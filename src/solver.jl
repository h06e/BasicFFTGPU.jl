export solve, solve!, effective_stiffness, HomogenizationResult

"""
    HomogenizationResult

Output of [`solve`](@ref). All tensors are in Kelvin notation
`(a11, a22, a33, √2 a23, √2 a13, √2 a12)`.

- `strain`, `stress`: local fields, `nx x ny x nz x 6` host arrays.
- `mean_strain`, `mean_stress`: macroscopic (volume averaged) tensors.
- `iterations`, `converged`, `residuals` (equilibrium residual per iteration).
"""
struct HomogenizationResult{T}
    strain::Array{T,4}
    stress::Array{T,4}
    mean_strain::Vector{Float64}
    mean_stress::Vector{Float64}
    iterations::Int
    converged::Bool
    residuals::Vector{Float64}
    grid::VoxelGrid
    scheme::Symbol
end

field_mean(a) = Float64(sum(a)) / length(a)
mean_dot(a, b) = sum(k -> Float64(real(dot(a[k], b[k]))), 1:6) / length(a[1])
means(f) = [field_mean(f[k]) for k in 1:6]

"""
    solve(material, loading; green=:continuous, device=Array, precision=Float32,
          tol=1e-6, maxiter=1000, accelerate=true, reference=nothing, verbose=false)

Solve the periodic linear elastic cell problem on `material`
([`PhaseMaterials`](@ref) or [`VoxelMaterials`](@ref)) under the macroscopic
`loading` ([`StrainLoading`](@ref), [`StressLoading`](@ref), [`MixedLoading`](@ref)).
See [`Workspace`](@ref) for `green`, `device`, `precision`, `reference`
and [`solve!`](@ref) for the remaining keywords.
"""
function solve(mat::MaterialDistribution, load::MacroLoading; green::Symbol=:continuous, device=Array,
               precision=Float32, reference=nothing, kwargs...)
    ws = Workspace(mat; green=green, device=device, precision=precision, reference=reference)
    return solve!(ws, load; kwargs...)
end

"""
    solve!(ws::Workspace, loading; tol=1e-6, maxiter=1000, accelerate=true,
           verbose=false, keep_fields=true) -> HomogenizationResult

Fixed-point (basic) scheme eps <- eps - rho * r, with the Barzilai-Borwein
step rho when `accelerate=true` (rho = 1 otherwise). The residual r is
Gamma0 * sigma(eps) for the fluctuation, plus, on stress-controlled
components, the mean correction -C0_ss^-1 (S_s - <sigma>_s) (the zero
frequency of the Green operator under stress control). Converged when

    sqrt(<|Gamma0 sigma|^2>) / |<eps>| < tol     (equilibrium)
    |<sigma>_s - S_s| / |<sigma>|      < tol     (imposed stress components s)

With `keep_fields=false`, only mean values are returned (empty field arrays).
"""
function solve!(ws::Workspace{T}, load::MacroLoading; tol::Real=1e-6, maxiter::Integer=1000,
                accelerate::Bool=true, verbose::Bool=false, keep_fields::Bool=true) where {T}
    ε, σ, R = ws.ε, ws.σ, ws.R
    s = findall(load.stress_controlled)          # stress-controlled components
    f = findall(.!collect(load.stress_controlled))  # strain-controlled components
    target = load.value
    C0 = reference_stiffness(ws)

    # Initial mean strain: imposed strains + reference-medium guess for the others
    E = zeros(6)
    E[f] .= target[f]
    if !isempty(s)
        E[s] .= C0[s, s] \ (target[s] .- C0[s, f] * E[f])
    end
    for k in 1:6
        fill!(ε[k], T(E[k]))
    end

    residuals = Float64[]
    converged = false
    rho = 1.0
    rsq_prev = 0.0
    ΔE_prev = zeros(length(s))
    it = 0

    compute_stress!(σ, ε, ws.material)
    while true
        Σ = means(σ)
        E = means(ε)
        # mean-strain correction on stress-controlled components
        ΔE = isempty(s) ? Float64[] : C0[s, s] \ (target[s] .- Σ[s])
        err_bc = isempty(s) ? 0.0 : norm(target[s] .- Σ[s]) / max(norm(Σ), norm(target), eps())

        apply_green!(ws)                          # σ <- Γ0 σ (zero mean)
        rsq_fluct = mean_dot(σ, σ)
        rsq = rsq_fluct + sum(abs2, ΔE)           # |r|^2, fluctuation and mean parts are orthogonal
        err_eq = sqrt(rsq_fluct) / max(norm(E), floatmin())
        push!(residuals, err_eq)
        verbose && @printf("iter %4d | equilibrium %.3e | stress bc %.3e | rho %.3f\n", it, err_eq, err_bc, rho)

        if (err_eq < tol && err_bc < tol) || norm(E) == 0 && norm(target) == 0
            converged = true
            break
        end
        it >= maxiter && break
        it += 1

        if accelerate && it > 1
            rho = rho / (1 - (mean_dot(σ, R) + dot(ΔE, ΔE_prev)) / rsq_prev)
        end
        rsq_prev = rsq
        ΔE_prev = ΔE

        for k in 1:6
            ε[k] .-= T(rho) .* σ[k]
        end
        for (j, k) in enumerate(s)
            ε[k] .+= T(rho * ΔE[j])
        end
        for k in 1:6
            copyto!(R[k], σ[k])
        end
        compute_stress!(σ, ε, ws.material)
    end
    converged || @warn "FFT solver did not converge in $maxiter iterations (residual $(last(residuals)))"

    compute_stress!(σ, ε, ws.material)           # σ was overwritten by the Green operator
    mean_strain = means(ε)
    mean_stress = means(σ)

    nx, ny, nz = ws.grid.size
    strain = keep_fields ? host_field(ε) : zeros(T, 0, 0, 0, 6)
    stress = keep_fields ? host_field(σ) : zeros(T, 0, 0, 0, 6)
    return HomogenizationResult{T}(strain, stress, mean_strain, mean_stress, it, converged,
                                   residuals, ws.grid, ws.scheme)
end

"Copy a Kelvin device field to a host nx x ny x nz x 6 array."
function host_field(f::NTuple{6})
    out = Array{eltype(f[1])}(undef, size(f[1])..., 6)
    for k in 1:6
        copyto!(view(out, :, :, :, k), Array(f[k]))
    end
    return out
end

"""
    effective_stiffness(material; kwargs...) -> 6x6 Matrix

Homogenized stiffness in Kelvin notation, from the 6 unit Kelvin macroscopic
strains. Keywords are passed to [`Workspace`](@ref) and [`solve!`](@ref).
"""
function effective_stiffness(mat::MaterialDistribution; green::Symbol=:continuous,
                             device=Array, precision=Float32, reference=nothing, kwargs...)
    ws = Workspace(mat; green=green, device=device, precision=precision, reference=reference)
    C = zeros(6, 6)
    for j in 1:6
        Ej = zeros(6)
        Ej[j] = 1
        r = solve!(ws, StrainLoading(Ej); keep_fields=false, kwargs...)
        C[:, j] .= r.mean_stress
    end
    return (C + C') / 2
end

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! VTK export of results

"Von Mises equivalent of a Kelvin stress vector: sqrt(3/2 s:s), s deviatoric."
function von_mises(s)
    p = (s[1] + s[2] + s[3]) / 3
    return sqrt(3 / 2 * ((s[1] - p)^2 + (s[2] - p)^2 + (s[3] - p)^2 + s[4]^2 + s[5]^2 + s[6]^2))
end

"""
    write_vtk(path, result::HomogenizationResult; material=nothing)

Write the local fields to a `.vti` file (cell data): `strain_kelvin` and
`stress_kelvin` (6-component Kelvin vectors), `strain` and `stress` as
ParaView symmetric tensors (tensor components, ParaView ordering XX YY ZZ XY YZ XZ,
for eigenvalues / principal directions), and `von_mises_stress`.
If `material` is given, the phase map (`phase`) or the voxel constants
(`kappa`, `mu`) are written too.
"""
function write_vtk(path::AbstractString, r::HomogenizationResult; material=nothing)
    isempty(r.strain) && throw(ArgumentError("result has no local fields (solved with keep_fields=false)"))
    fields = Dict{String,Array}(
        "strain_kelvin" => permutedims(r.strain, (4, 1, 2, 3)),
        "stress_kelvin" => permutedims(r.stress, (4, 1, 2, 3)),
        "strain" => paraview_tensor(r.strain),
        "stress" => paraview_tensor(r.stress),
        "von_mises_stress" => [von_mises(view(r.stress, i, j, k, :)) for i in axes(r.stress, 1),
                               j in axes(r.stress, 2), k in axes(r.stress, 3)],
    )
    if material isa PhaseMaterials
        fields["phase"] = phase_labels(material)
    elseif material isa VoxelMaterials
        fields["kappa"] = material.kappa
        fields["mu"] = material.mu
    end
    return write_vtk(path, fields, r.grid)
end
