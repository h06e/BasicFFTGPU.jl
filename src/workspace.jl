export Workspace

"""
    Workspace(material; green=:continuous, device=Array, precision=Float32, reference=nothing)

Pre-allocated buffers (on `device`) for solving problems on one material
distribution. Reused across loadings, e.g. by [`effective_stiffness`](@ref).

- `green`: `:continuous` (Moulinec & Suquet continuous frequencies, default) or
  `:staggered` (staggered grid). Aliases: `:moulinec_suquet`, `:MS`, `:SG`.
- `device`: array constructor, `Array` (CPU, FFTW) or `CuArray` (GPU, after `using CUDA`).
- `precision`: `Float32` or `Float64`.
- `reference`: `(kappa0, mu0)` of the reference medium; default from the material bounds.
"""
struct Workspace{T,F,C,PF,PI,X,M}
    grid::VoxelGrid
    scheme::Symbol
    ε::NTuple{6,F}
    σ::NTuple{6,F}
    R::NTuple{6,F}       # previous Green residual (Barzilai-Borwein step)
    τ::NTuple{6,C}       # Fourier-space scratch
    P::PF
    Pinv::PI
    ξ::X
    nyq::NTuple{3,Int}
    material::M
    kappa0::T
    mu0::T
end

function Workspace(mat::MaterialDistribution; green::Symbol=:continuous, device=Array,
                   precision::Type{T}=Float32, reference=nothing) where {T<:AbstractFloat}
    scheme = green_scheme(green)
    grid = mat.grid
    n = grid.size
    nc = (n[1] ÷ 2 + 1, n[2], n[3])

    ε = ntuple(_ -> device(zeros(T, n)), 6)
    σ = ntuple(_ -> device(zeros(T, n)), 6)
    R = ntuple(_ -> device(zeros(T, n)), 6)
    τ = ntuple(_ -> device(zeros(Complex{T}, nc)), 6)

    P = plan_rfft(ε[1])
    Pinv = plan_irfft(τ[1], n[1])

    ξ = map(device, green_frequencies(scheme, grid, T))

    kappa0, mu0 = reference === nothing ? reference_medium(mat) : reference
    return Workspace(grid, scheme, ε, σ, R, τ, P, Pinv, ξ, nyquist_indices(n),
                     to_device(mat, device, T), T(kappa0), T(mu0))
end

"Reference stiffness (Kelvin, host, Float64)."
reference_stiffness(ws::Workspace) = isotropic_stiffness(Float64(ws.kappa0), Float64(ws.mu0))

"In place: `ws.σ` <- Gamma0 * `ws.σ`."
function apply_green!(ws::Workspace)
    for k in 1:6
        mul!(ws.τ[k], ws.P, ws.σ[k])
    end

    mu0 = ws.mu0
    lambda0 = ws.kappa0 - 2mu0 / 3
    coef = (lambda0 + mu0) / (mu0 * (lambda0 + 2mu0))
    backend = get_backend(ws.τ[1])
    if ws.scheme === :continuous
        green_continuous_kernel!(backend)(ws.τ, ws.ξ..., ws.nyq, mu0, coef; ndrange=size(ws.τ[1]))
    else
        green_staggered_kernel!(backend)(ws.τ, ws.ξ..., mu0, coef; ndrange=size(ws.τ[1]))
    end

    for k in 1:6
        mul!(ws.σ[k], ws.Pinv, ws.τ[k])
    end
    return nothing
end
