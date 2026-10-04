#! Constitutive relation sigma = C(x) : eps, on Kelvin fields (device side).

@inline function iso_stress!(σ, ε, I, kappa, mu)
    @inbounds begin
        lambda = kappa - 2mu / 3
        tr = ε[1][I] + ε[2][I] + ε[3][I]
        σ[1][I] = lambda * tr + 2mu * ε[1][I]
        σ[2][I] = lambda * tr + 2mu * ε[2][I]
        σ[3][I] = lambda * tr + 2mu * ε[3][I]
        σ[4][I] = 2mu * ε[4][I]
        σ[5][I] = 2mu * ε[5][I]
        σ[6][I] = 2mu * ε[6][I]
    end
    return nothing
end

"Isotropic phases: `props[1, p]`, `props[2, p]` = kappa, mu of phase p."
@kernel function stress_iso_phase_kernel!(σ, @Const(ε), @Const(index), @Const(props))
    I = @index(Global, Linear)
    @inbounds p = index[I]
    @inbounds iso_stress!(σ, ε, I, props[1, p], props[2, p])
end

"Isotropic voxel-wise constants."
@kernel function stress_iso_voxel_kernel!(σ, @Const(ε), @Const(kappa), @Const(mu))
    I = @index(Global, Linear)
    @inbounds iso_stress!(σ, ε, I, kappa[I], mu[I])
end

"General anisotropic phases: `props[:, p]` = Kelvin stiffness of phase p, column-major 6x6."
@kernel function stress_aniso_phase_kernel!(σ, @Const(ε), @Const(index), @Const(props))
    I = @index(Global, Linear)
    @inbounds begin
        p = index[I]
        e1 = ε[1][I]; e2 = ε[2][I]; e3 = ε[3][I]; e4 = ε[4][I]; e5 = ε[5][I]; e6 = ε[6][I]
        Base.Cartesian.@nexprs 6 i -> begin
            σ[i][I] = props[i, p] * e1 + props[i+6, p] * e2 + props[i+12, p] * e3 +
                      props[i+18, p] * e4 + props[i+24, p] * e5 + props[i+30, p] * e6
        end
    end
end

#! Device-side material data, built once per workspace.

abstract type DeviceMaterial end

struct DeviceIsoPhases{A,P} <: DeviceMaterial
    index::A
    props::P
end

struct DeviceAnisoPhases{A,P} <: DeviceMaterial
    index::A
    props::P
end

struct DeviceIsoVoxels{F} <: DeviceMaterial
    kappa::F
    mu::F
end

function to_device(m::PhaseMaterials, device, ::Type{T}) where {T}
    index = device(m.index)
    if all(x -> x isa Isotropic, m.materials)
        props = vcat(T[x.kappa for x in m.materials]', T[x.mu for x in m.materials]')
        return DeviceIsoPhases(index, device(props))
    end
    props = Matrix{T}(undef, 36, length(m.materials))
    for (p, x) in enumerate(m.materials)
        props[:, p] .= vec(stiffness(x))
    end
    return DeviceAnisoPhases(index, device(props))
end

to_device(m::VoxelMaterials, device, ::Type{T}) where {T} =
    DeviceIsoVoxels(device(T.(m.kappa)), device(T.(m.mu)))

function compute_stress!(σ, ε, m::DeviceIsoPhases)
    backend = get_backend(σ[1])
    stress_iso_phase_kernel!(backend)(σ, ε, m.index, m.props; ndrange=length(σ[1]))
end

function compute_stress!(σ, ε, m::DeviceAnisoPhases)
    backend = get_backend(σ[1])
    stress_aniso_phase_kernel!(backend)(σ, ε, m.index, m.props; ndrange=length(σ[1]))
end

function compute_stress!(σ, ε, m::DeviceIsoVoxels)
    backend = get_backend(σ[1])
    stress_iso_voxel_kernel!(backend)(σ, ε, m.kappa, m.mu; ndrange=length(σ[1]))
end
