#! Green operator of the isotropic reference medium (kappa0, mu0), applied in
#! Fourier space to a Kelvin 6-component field. Two discretizations:
#!
#! - :continuous  Moulinec & Suquet (1994/1998), continuous frequencies xi = k / (N h).
#! - :staggered   staggered grid, modified frequencies xi = (exp(-2i pi k/N) - 1) / h
#!                (finite-difference consistent; same scheme as Kairotop).
#!
#! Both kernels map tau -> Gamma0 * tau in place. The zero frequency is
#! zeroed (mean imposed separately by the solver). Nyquist frequencies of
#! even-sized directions are zeroed for :continuous only, where the
#! symbol is not Hermitian; the staggered symbol is well defined there.

const GREEN_SCHEMES = (:continuous, :staggered)

"Normalize user aliases to one of GREEN_SCHEMES."
function green_scheme(s::Symbol)
    s in (:continuous, :moulinec_suquet, :MS) && return :continuous
    s in (:staggered, :staggered_grid, :SG) && return :staggered
    throw(ArgumentError("unknown Green operator `$s`; use :continuous (Moulinec-Suquet) or :staggered"))
end

"Frequency vectors (host) for an rfft along dim 1 and full fft along dims 2, 3."
function green_frequencies(scheme::Symbol, grid::VoxelGrid, ::Type{T}) where {T}
    n = grid.size
    h = grid.spacing
    k1 = rfftfreq(n[1], n[1])
    k2 = fftfreq(n[2], n[2])
    k3 = fftfreq(n[3], n[3])
    ks = (k1, k2, k3)
    if scheme === :continuous
        return ntuple(d -> T.(ks[d] ./ (n[d] * h[d])), 3)
    else
        return ntuple(d -> Complex{T}.((exp.(-2im * pi .* ks[d] ./ n[d]) .- 1) ./ h[d]), 3)
    end
end

"1-based index of the Nyquist frequency along each direction (0 if n is odd)."
nyquist_indices(n) = ntuple(d -> iseven(n[d]) ? n[d] ÷ 2 + 1 : 0, 3)

@kernel function green_continuous_kernel!(τ, @Const(ξ1), @Const(ξ2), @Const(ξ3), nyq, mu0, coef)
    i1, i2, i3 = @index(Global, NTuple)
    T = real(eltype(τ[1]))
    is2 = inv(sqrt(T(2)))
    @inbounds begin
        if (i1 == 1 && i2 == 1 && i3 == 1) || i1 == nyq[1] || i2 == nyq[2] || i3 == nyq[3]
            for k in 1:6
                τ[k][i1, i2, i3] = zero(eltype(τ[1]))
            end
        else
            x1 = ξ1[i1]
            x2 = ξ2[i2]
            x3 = ξ3[i3]
            t1 = τ[1][i1, i2, i3]; t2 = τ[2][i1, i2, i3]; t3 = τ[3][i1, i2, i3]
            t4 = τ[4][i1, i2, i3] * is2; t5 = τ[5][i1, i2, i3] * is2; t6 = τ[6][i1, i2, i3] * is2

            # tau . xi
            d1 = x1 * t1 + x2 * t6 + x3 * t5
            d2 = x1 * t6 + x2 * t2 + x3 * t4
            d3 = x1 * t5 + x2 * t4 + x3 * t3

            xs = x1 * x1 + x2 * x2 + x3 * x3
            dd = x1 * d1 + x2 * d2 + x3 * d3

            f1 = d1 / (xs * mu0) - x1 * coef * dd / (xs * xs)
            f2 = d2 / (xs * mu0) - x2 * coef * dd / (xs * xs)
            f3 = d3 / (xs * mu0) - x3 * coef * dd / (xs * xs)

            τ[1][i1, i2, i3] = x1 * f1
            τ[2][i1, i2, i3] = x2 * f2
            τ[3][i1, i2, i3] = x3 * f3
            τ[4][i1, i2, i3] = (x2 * f3 + x3 * f2) * is2
            τ[5][i1, i2, i3] = (x1 * f3 + x3 * f1) * is2
            τ[6][i1, i2, i3] = (x1 * f2 + x2 * f1) * is2
        end
    end
end

@kernel function green_staggered_kernel!(τ, @Const(ξ1), @Const(ξ2), @Const(ξ3), mu0, coef)
    i1, i2, i3 = @index(Global, NTuple)
    T = real(eltype(τ[1]))
    is2 = inv(sqrt(T(2)))
    @inbounds begin
        if i1 == 1 && i2 == 1 && i3 == 1
            for k in 1:6
                τ[k][i1, i2, i3] = zero(eltype(τ[1]))
            end
        else
            x1 = ξ1[i1]
            x2 = ξ2[i2]
            x3 = ξ3[i3]
            c1 = conj(x1); c2 = conj(x2); c3 = conj(x3)
            t1 = τ[1][i1, i2, i3]; t2 = τ[2][i1, i2, i3]; t3 = τ[3][i1, i2, i3]
            t4 = τ[4][i1, i2, i3] * is2; t5 = τ[5][i1, i2, i3] * is2; t6 = τ[6][i1, i2, i3] * is2

            # discrete divergence: normal stresses at voxel centers, shear stresses on edges
            d1 = -x1 * t1 + c2 * t6 + c3 * t5
            d2 = c1 * t6 - x2 * t2 + c3 * t4
            d3 = c1 * t5 + c2 * t4 - x3 * t3

            xs = c1 * x1 + c2 * x2 + c3 * x3
            dd = c1 * d1 + c2 * d2 + c3 * d3

            f1 = d1 / (xs * mu0) - x1 * coef * dd / (xs * xs)
            f2 = d2 / (xs * mu0) - x2 * coef * dd / (xs * xs)
            f3 = d3 / (xs * mu0) - x3 * coef * dd / (xs * xs)

            τ[1][i1, i2, i3] = -c1 * f1
            τ[2][i1, i2, i3] = -c2 * f2
            τ[3][i1, i2, i3] = -c3 * f3
            τ[4][i1, i2, i3] = (x2 * f3 + x3 * f2) * is2
            τ[5][i1, i2, i3] = (x1 * f3 + x3 * f1) * is2
            τ[6][i1, i2, i3] = (x1 * f2 + x2 * f1) * is2
        end
    end
end
