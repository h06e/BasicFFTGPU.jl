export kelvin, kelvin_to_tensor, isotropic_stiffness, cubic_stiffness, rotation_kelvin, rotate_stiffness

#! Kelvin notation everywhere (user side and device side).
#!
#! A symmetric second-order tensor a is the 6-vector
#!     (a11, a22, a33, sqrt(2) a23, sqrt(2) a13, sqrt(2) a12)
#! and a fourth-order tensor with minor symmetries is the 6x6 matrix acting on
#! it. The basis is orthonormal: double contractions are plain dot products
#! and C : eps is the matrix-vector product (see docs/CONVENTIONS.md).

const KELVIN_SHEAR = ((2, 3), (1, 3), (1, 2))

"""
    kelvin(a::AbstractMatrix) -> Vector

Kelvin 6-vector of a symmetric 3x3 tensor:
`(a11, a22, a33, √2 a23, √2 a13, √2 a12)`.
"""
function kelvin(a::AbstractMatrix)
    size(a) == (3, 3) || throw(ArgumentError("expected a 3x3 matrix, got size $(size(a))"))
    isapprox(a, a'; rtol=1e-12, atol=1e-14) || throw(ArgumentError("tensor must be symmetric"))
    s = sqrt(2.0)
    return Float64[a[1, 1], a[2, 2], a[3, 3], s * a[2, 3], s * a[1, 3], s * a[1, 2]]
end

"""
    kelvin_to_tensor(v) -> 3x3 Matrix

Symmetric 3x3 tensor from its Kelvin 6-vector (inverse of [`kelvin`](@ref)).
"""
function kelvin_to_tensor(v::AbstractVector)
    length(v) == 6 || throw(ArgumentError("expected a Kelvin 6-vector"))
    a = zeros(3, 3)
    for k in 1:3
        a[k, k] = v[k]
    end
    for (k, (i, j)) in enumerate(KELVIN_SHEAR)
        a[i, j] = a[j, i] = v[k+3] / sqrt(2.0)
    end
    return a
end

"""
    isotropic_stiffness(kappa, mu) -> 6x6 Matrix

Isotropic stiffness in Kelvin notation, `3 kappa J + 2 mu K`.
"""
function isotropic_stiffness(kappa, mu)
    lambda = kappa - 2mu / 3
    C = zeros(6, 6)
    C[1:3, 1:3] .= lambda
    for k in 1:6
        C[k, k] += 2mu
    end
    return C
end

"Kelvin 6-vector from a 6-vector (taken as Kelvin) or a symmetric 3x3 matrix."
function as_kelvin(v::Union{AbstractVector,Tuple})
    length(v) == 6 || throw(ArgumentError("expected a Kelvin 6-vector, got $(length(v)) components"))
    return Float64.(collect(v))
end
as_kelvin(m::AbstractMatrix) = kelvin(m)

"""
    cubic_stiffness(c11, c12, c44) -> 6x6 Matrix

Cubic stiffness (crystal axes along x, y, z) in Kelvin notation, from the
tensor components c11 = C1111, c12 = C1122, c44 = C2323.
"""
function cubic_stiffness(c11, c12, c44)
    C = zeros(6, 6)
    C[1:3, 1:3] .= c12
    for k in 1:3
        C[k, k] = c11
        C[k+3, k+3] = 2c44
    end
    return C
end

"""
    rotation_kelvin(R) -> 6x6 Matrix

Orthogonal 6x6 matrix Q such that `kelvin(R * a * R') == Q * kelvin(a)` for
any symmetric `a`, with `R` a 3x3 rotation matrix.
"""
function rotation_kelvin(R::AbstractMatrix)
    size(R) == (3, 3) || throw(ArgumentError("expected a 3x3 rotation matrix"))
    Q = zeros(6, 6)
    for j in 1:6
        e = zeros(6)
        e[j] = 1
        Q[:, j] .= kelvin(R * kelvin_to_tensor(e) * R')
    end
    return Q
end

"""
    rotate_stiffness(C, R) -> 6x6 Matrix

Kelvin stiffness of a material with stiffness `C` (Kelvin) rotated by `R`:
`Q * C * Q'` with `Q = rotation_kelvin(R)`.
"""
function rotate_stiffness(C::AbstractMatrix, R::AbstractMatrix)
    Q = rotation_kelvin(R)
    return Q * C * Q'
end
