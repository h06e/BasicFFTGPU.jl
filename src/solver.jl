export solve, solve!, effective_stiffness, HomogenizationResult, commit!, plastic_strain, cumulated_plastic_strain

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
           verbose=false, keep_fields=true, warm_start=false) -> HomogenizationResult

Fixed-point (basic) scheme eps <- eps - rho * r, with the Barzilai-Borwein
step rho when `accelerate=true` (rho = 1 otherwise). The residual r is
Gamma0 * sigma(eps) for the fluctuation, plus, on stress-controlled
components, the mean correction -C0_ss^-1 (S_s - <sigma>_s) (the zero
frequency of the Green operator under stress control). Converged when

    sqrt(<|Gamma0 sigma|^2>) / |<eps>| < tol     (equilibrium)
    |<sigma>_s - S_s| / |<sigma>|      < tol     (imposed stress components s)

With `keep_fields=false`, only mean values are returned (empty field arrays).

`warm_start=true` starts from the strain field left in `ws` by the previous solve,
shifted to the new imposed mean strain: use it for load increments on a
history-dependent material ([`VoxelPlasticMaterials`](@ref)), together with
[`commit!`](@ref). For such materials the mean strain may vanish (unloading), so
the equilibrium residual is normalized by the rms strain `sqrt(<eps:eps>)` instead of `|<eps>|`.
"""
function solve!(ws::Workspace{T}, load::MacroLoading; tol::Real=1e-6, maxiter::Integer=1000,
                accelerate::Bool=true, verbose::Bool=false, keep_fields::Bool=true,
                warm_start::Bool=false) where {T}
    ε, σ, R = ws.ε, ws.σ, ws.R
    s = findall(load.stress_controlled)          # stress-controlled components
    f = findall(.!collect(load.stress_controlled))  # strain-controlled components
    target = load.value
    C0 = reference_stiffness(ws)

    if warm_start
        # previous strain field, shifted to the imposed mean strain components
        E = means(ε)
        for k in f
            ε[k] .+= T(target[k] - E[k])
        end
    else
        # Initial mean strain: imposed strains + reference-medium guess for the others
        E = zeros(6)
        E[f] .= target[f]
        if !isempty(s)
            E[s] .= C0[s, s] \ (target[s] .- C0[s, f] * E[f])
        end
        for k in 1:6
            fill!(ε[k], T(E[k]))
        end
    end
    history = ws.material isa DevicePlasticVoxels

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
        strain_scale = history ? sqrt(mean_dot(ε, ε)) : norm(E)
        # mean-strain correction on stress-controlled components
        ΔE = isempty(s) ? Float64[] : C0[s, s] \ (target[s] .- Σ[s])
        err_bc = isempty(s) ? 0.0 : norm(target[s] .- Σ[s]) / max(norm(Σ), norm(target), eps())

        apply_green!(ws)                          # σ <- Γ0 σ (zero mean)
        rsq_fluct = mean_dot(σ, σ)
        rsq = rsq_fluct + sum(abs2, ΔE)           # |r|^2, fluctuation and mean parts are orthogonal
        err_eq = sqrt(rsq_fluct) / max(strain_scale, floatmin())
        push!(residuals, err_eq)
        verbose && @printf("iter %4d | equilibrium %.3e | stress bc %.3e | rho %.3f\n", it, err_eq, err_bc, rho)

        if err_eq < tol && err_bc < tol
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

"""
    commit!(ws::Workspace)

End of a converged load increment: the internal state (plastic strain, cumulated
plastic strain) of the last solve becomes the reference state of the next
increment. No-op for elastic materials.
"""
commit!(ws::Workspace) = commit_state!(ws.material)

"""
    plastic_strain(ws::Workspace) -> nx x ny x nz x 6 Array (Kelvin)
    cumulated_plastic_strain(ws::Workspace) -> nx x ny x nz Array

Plastic strain and cumulated plastic strain `p` of the last solve on `ws`
(material [`VoxelPlasticMaterials`](@ref)), copied to the host.
"""
plastic_strain(ws::Workspace) = host_field(plastic_state(ws.material).εp)
cumulated_plastic_strain(ws::Workspace) = Array(plastic_state(ws.material).p)
plastic_state(m::DevicePlasticVoxels) = m
plastic_state(::DeviceMaterial) = throw(ArgumentError("the material of this workspace has no plastic state"))

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
    write_vtk(path, result::HomogenizationResult; material=nothing, workspace=nothing, fields=Dict(),
              compress=false)

Write the local fields to a `.vti` file (cell data): one scalar field per tensor
component (`strain_11`, `strain_22`, `strain_33`, `strain_23`, `strain_13`, `strain_12`,
same for `stress`; plain tensor components, not Kelvin), `strain_kelvin` and
`stress_kelvin` (6-component Kelvin vectors), `strain` and `stress` as
ParaView symmetric tensors (tensor components, ParaView ordering XX YY ZZ XY YZ XZ,
for eigenvalues / principal directions), and `von_mises_stress`.
If `material` is given, the phase map (`phase`) or the voxel constants
(`kappa`, `mu`, and `sigma0`, `Q` for plastic materials) are written too. With the
`workspace` of a plastic material: `plastic_strain_ij` (components),
`plastic_strain` (ParaView tensor), `plastic_strain_kelvin` and `cumulated_plastic_strain`. `fields`: extra
`name => nx x ny x nz array` cell fields.
"""
function write_vtk(path::AbstractString, r::HomogenizationResult; material=nothing, workspace=nothing,
                   fields=Dict(), compress=false)
    isempty(r.strain) && throw(ArgumentError("result has no local fields (solved with keep_fields=false)"))
    out = Dict{String,Array}(
        "strain_kelvin" => permutedims(r.strain, (4, 1, 2, 3)),
        "stress_kelvin" => permutedims(r.stress, (4, 1, 2, 3)),
        "strain" => paraview_tensor(r.strain),
        "stress" => paraview_tensor(r.stress),
        "von_mises_stress" => [von_mises(view(r.stress, i, j, k, :)) for i in axes(r.stress, 1),
                               j in axes(r.stress, 2), k in axes(r.stress, 3)],
    )
    tensor_components!(out, "strain", r.strain)
    tensor_components!(out, "stress", r.stress)
    if material isa PhaseMaterials
        out["phase"] = phase_labels(material)
    elseif material isa Union{VoxelMaterials,VoxelPlasticMaterials}
        out["kappa"] = material.kappa
        out["mu"] = material.mu
    end
    if material isa VoxelPlasticMaterials
        out["sigma0"] = material.sigma0
        out["Q"] = material.Q
    end
    if workspace !== nothing && workspace.material isa DevicePlasticVoxels
        εp = plastic_strain(workspace)
        out["plastic_strain_kelvin"] = permutedims(εp, (4, 1, 2, 3))
        out["plastic_strain"] = paraview_tensor(εp)
        tensor_components!(out, "plastic_strain", εp)
        out["cumulated_plastic_strain"] = cumulated_plastic_strain(workspace)
    end
    for (name, f) in fields
        out[String(name)] = f
    end
    return write_vtk(path, out, r.grid; compress=compress)
end
