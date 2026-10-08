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

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! J2 elastoplasticity, isotropic hardening R(p) = sigma0 + H p + Q (1 - exp(-b p))
#!
#! Radial return from the committed state (εp_n, p_n): the trial stress is
#! elastic, and if its von Mises stress q exceeds R(p_n), the plastic multiplier
#! dp solves q - 3 mu dp = R(p_n + dp) (scalar Newton, monotone: R is concave).
#! The trial state (εp, p) is overwritten at every call; commit! makes it the
#! committed state at the end of a converged increment.

@kernel function stress_plastic_voxel_kernel!(σ, εp, p, @Const(ε), @Const(εp_n), @Const(p_n),
                                              @Const(kappa), @Const(mu), @Const(sigma0),
                                              @Const(Q), @Const(b), @Const(H))
    I = @index(Global, Linear)
    T = eltype(σ[1])
    @inbounds begin
        k = kappa[I]; m = mu[I]
        e1 = ε[1][I] - εp_n[1][I]; e2 = ε[2][I] - εp_n[2][I]; e3 = ε[3][I] - εp_n[3][I]
        e4 = ε[4][I] - εp_n[4][I]; e5 = ε[5][I] - εp_n[5][I]; e6 = ε[6][I] - εp_n[6][I]
        tr = e1 + e2 + e3
        # trial deviatoric stress s = 2 mu dev(εe) (Kelvin components)
        s1 = 2m * (e1 - tr / 3); s2 = 2m * (e2 - tr / 3); s3 = 2m * (e3 - tr / 3)
        s4 = 2m * e4; s5 = 2m * e5; s6 = 2m * e6
        q = sqrt(T(1.5) * (s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5 + s6 * s6))

        pn = p_n[I]; s0 = sigma0[I]; q_ = Q[I]; b_ = b[I]; h = H[I]
        dp = zero(T)
        if q > s0 + h * pn + q_ * (1 - exp(-b_ * pn))
            for _ in 1:50
                pp = pn + dp
                ex = exp(-b_ * pp)
                g = q - 3m * dp - (s0 + h * pp + q_ * (1 - ex))
                δ = g / (3m + h + q_ * b_ * ex)
                dp += δ
                abs(δ) <= 4eps(T) * dp && break
            end
        end
        # s = s_trial (1 - 3 mu dp / q), Δεp = dp * 3/2 s_trial / q
        a = dp > 0 ? 3m * dp / q : zero(T)
        c = dp > 0 ? T(1.5) * dp / q : zero(T)
        pm = k * tr
        σ[1][I] = pm + s1 * (1 - a); σ[2][I] = pm + s2 * (1 - a); σ[3][I] = pm + s3 * (1 - a)
        σ[4][I] = s4 * (1 - a); σ[5][I] = s5 * (1 - a); σ[6][I] = s6 * (1 - a)
        εp[1][I] = εp_n[1][I] + c * s1; εp[2][I] = εp_n[2][I] + c * s2; εp[3][I] = εp_n[3][I] + c * s3
        εp[4][I] = εp_n[4][I] + c * s4; εp[5][I] = εp_n[5][I] + c * s5; εp[6][I] = εp_n[6][I] + c * s6
        p[I] = pn + dp
    end
end

struct DevicePlasticVoxels{F} <: DeviceMaterial
    kappa::F
    mu::F
    sigma0::F
    Q::F
    b::F
    H::F
    εp::NTuple{6,F}     # trial state (last stress evaluation)
    p::F
    εp_n::NTuple{6,F}   # committed state (end of the last converged increment)
    p_n::F
end

function to_device(m::VoxelPlasticMaterials, device, ::Type{T}) where {T}
    d(a) = device(T.(a))
    z() = device(zeros(T, m.grid.size))
    return DevicePlasticVoxels(d(m.kappa), d(m.mu), d(m.sigma0), d(m.Q), d(m.b), d(m.H),
                               ntuple(_ -> z(), 6), z(), ntuple(_ -> z(), 6), z())
end

function compute_stress!(σ, ε, m::DevicePlasticVoxels)
    backend = get_backend(σ[1])
    stress_plastic_voxel_kernel!(backend)(σ, m.εp, m.p, ε, m.εp_n, m.p_n, m.kappa, m.mu, m.sigma0,
                                          m.Q, m.b, m.H; ndrange=length(σ[1]))
end

"Make the trial internal state the committed one (no-op for elastic materials)."
commit_state!(::DeviceMaterial) = nothing
function commit_state!(m::DevicePlasticVoxels)
    for k in 1:6
        copyto!(m.εp_n[k], m.εp[k])
    end
    copyto!(m.p_n, m.p)
    return nothing
end
