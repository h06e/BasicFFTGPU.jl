export VoxelGrid, Isotropic, Anisotropic, PhaseMaterials, VoxelMaterials, VoxelPlasticMaterials, gridsize

"""
    VoxelGrid(size; spacing=(1,1,1), origin=(0,0,0))

Geometry of a periodic voxel grid: number of voxels per direction, voxel size
and position of the lower corner of the first voxel. Use `size = (nx, ny, 1)`
for 2D (plane strain) problems.
"""
struct VoxelGrid
    size::NTuple{3,Int}
    spacing::NTuple{3,Float64}
    origin::NTuple{3,Float64}
end
VoxelGrid(size; spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)) =
    VoxelGrid(Tuple(Int.(size)), Tuple(Float64.(spacing)), Tuple(Float64.(origin)))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Single materials

abstract type Material end

"""
    Isotropic(; kappa, mu)
    Isotropic(; E, nu)
    Isotropic(; lambda, mu)

Isotropic linear elastic material. Stored as bulk (`kappa`) and shear (`mu`) moduli.
"""
struct Isotropic <: Material
    kappa::Float64
    mu::Float64
end

function Isotropic(; kappa=nothing, mu=nothing, E=nothing, nu=nothing, lambda=nothing)
    k, m = isotropic_moduli(kappa, mu, E, nu, lambda)
    return Isotropic(k, m)
end

"Convert any supported pair of isotropic constants (scalars or arrays) to (kappa, mu)."
function isotropic_moduli(kappa, mu, E, nu, lambda)
    given = (kappa !== nothing, mu !== nothing, E !== nothing, nu !== nothing, lambda !== nothing)
    if given == (true, true, false, false, false)
        return kappa, mu
    elseif given == (false, false, true, true, false)
        return (@. E / (3 * (1 - 2nu))), (@. E / (2 * (1 + nu)))
    elseif given == (false, true, false, false, true)
        return (@. lambda + 2mu / 3), mu
    end
    throw(ArgumentError("give exactly one of the pairs (kappa, mu), (E, nu) or (lambda, mu)"))
end

"""
    Anisotropic(C)

General (anisotropic) linear elastic material from its 6x6 stiffness matrix in
Kelvin notation (see [`kelvin`](@ref)): `sigma = C * eps` on Kelvin 6-vectors.
"""
struct Anisotropic <: Material
    C::Matrix{Float64}   # Kelvin notation
    function Anisotropic(C::AbstractMatrix)
        size(C) == (6, 6) || throw(ArgumentError("stiffness matrix must be 6x6"))
        Cm = Matrix{Float64}(C)
        isapprox(Cm, Cm'; rtol=1e-10) || throw(ArgumentError("stiffness matrix must be symmetric"))
        isposdef(Symmetric(Cm)) || throw(ArgumentError("stiffness matrix must be positive definite"))
        return new(Cm)
    end
end

stiffness(m::Isotropic) = isotropic_stiffness(m.kappa, m.mu)
stiffness(m::Anisotropic) = m.C

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Material distributions over the grid

abstract type MaterialDistribution end

"""
    PhaseMaterials(phases, materials; grid=VoxelGrid(size(phases)))

Material given by a phase map plus a (short) list of materials.

- `phases`: 3D integer array of phase labels (labels can start at 0).
- `materials`: either a `Dict`/list of pairs `label => material`, or a vector
  of materials indexed by label (labels must then be `1:length(materials)`).
"""
struct PhaseMaterials <: MaterialDistribution
    index::Array{Int32,3}       # 1-based index into `materials`
    labels::Vector{Int}         # original label of each material
    materials::Vector{Material}
    grid::VoxelGrid
end

function PhaseMaterials(phases::AbstractArray{<:Integer,3}, materials::AbstractDict{<:Integer,<:Material};
                        grid::VoxelGrid=VoxelGrid(size(phases)))
    grid.size == size(phases) || throw(DimensionMismatch("grid size $(grid.size) != phase map size $(size(phases))"))
    labels = sort!(collect(keys(materials)))
    missing_labels = setdiff(unique(phases), labels)
    isempty(missing_labels) || throw(ArgumentError("no material given for phase label(s) $(missing_labels)"))
    lookup = Dict(l => Int32(k) for (k, l) in enumerate(labels))
    index = map(p -> lookup[p], phases)
    return PhaseMaterials(index, labels, Material[materials[l] for l in labels], grid)
end

PhaseMaterials(phases::AbstractArray{<:Integer,3}, materials::AbstractVector{<:Material}; kwargs...) =
    PhaseMaterials(phases, Dict(k => m for (k, m) in enumerate(materials)); kwargs...)

PhaseMaterials(phases::AbstractArray{<:Integer,3}, materials::Pair...; kwargs...) =
    PhaseMaterials(phases, Dict{Int,Material}(materials...); kwargs...)

"Phase labels as given by the user (inverse of the internal 1-based indexing)."
phase_labels(m::PhaseMaterials) = map(i -> m.labels[i], m.index)

"""
    VoxelMaterials(; kappa, mu, grid=...)
    VoxelMaterials(; E, nu, grid=...)
    VoxelMaterials(; lambda, mu, grid=...)

Isotropic material whose elastic constants are given voxel per voxel, as 3D arrays.
"""
struct VoxelMaterials <: MaterialDistribution
    kappa::Array{Float64,3}
    mu::Array{Float64,3}
    grid::VoxelGrid
end

function VoxelMaterials(; kappa=nothing, mu=nothing, E=nothing, nu=nothing, lambda=nothing, grid=nothing)
    k, m = isotropic_moduli(kappa, mu, E, nu, lambda)
    size(k) == size(m) || throw(DimensionMismatch("elastic constant fields have different sizes"))
    ndims(k) == 3 || throw(ArgumentError("elastic constant fields must be 3D arrays"))
    g = grid === nothing ? VoxelGrid(size(k)) : grid
    g.size == size(k) || throw(DimensionMismatch("grid size $(g.size) != field size $(size(k))"))
    (all(>(0), k) && all(>(0), m)) || throw(ArgumentError("bulk and shear moduli must be positive in every voxel"))
    return VoxelMaterials(Float64.(k), Float64.(m), g)
end

"""
    VoxelPlasticMaterials(; E, nu, sigma0, Q=0, b=0, H=0, grid=...)

Voxel-wise J2 (von Mises) elastoplasticity with isotropic hardening, small strains.
Elasticity is given as for [`VoxelMaterials`](@ref) (`kappa, mu`, `E, nu` or `lambda, mu`).
The yield stress depends on the cumulated plastic strain `p` (Voce law plus a linear term):

    R(p) = sigma0 + H p + Q (1 - exp(-b p))

Each constant is a 3D array or a scalar (same value in every voxel). The material has a
history: solve successive load increments on one [`Workspace`](@ref) and call
[`commit!`](@ref) after each converged increment.
"""
struct VoxelPlasticMaterials <: MaterialDistribution
    kappa::Array{Float64,3}
    mu::Array{Float64,3}
    sigma0::Array{Float64,3}
    Q::Array{Float64,3}
    b::Array{Float64,3}
    H::Array{Float64,3}
    grid::VoxelGrid
end

function VoxelPlasticMaterials(; kappa=nothing, mu=nothing, E=nothing, nu=nothing, lambda=nothing,
                               sigma0, Q=0.0, b=0.0, H=0.0, grid=nothing)
    arrays = filter(x -> x isa AbstractArray, [kappa, mu, E, nu, lambda, sigma0, Q, b, H])
    grid === nothing && isempty(arrays) && throw(ArgumentError("give `grid` when every constant is a scalar"))
    sz = grid === nothing ? size(first(arrays)) : grid.size
    field(x, name) = if x isa Real
        fill(Float64(x), sz)
    else
        size(x) == sz || throw(DimensionMismatch("$name has size $(size(x)), expected $sz"))
        Float64.(x)
    end
    opt(x, name) = x === nothing ? nothing : field(x, name)
    el = VoxelMaterials(; kappa=opt(kappa, "kappa"), mu=opt(mu, "mu"), E=opt(E, "E"), nu=opt(nu, "nu"),
                        lambda=opt(lambda, "lambda"), grid=grid)
    s0, q, bb, h = field(sigma0, "sigma0"), field(Q, "Q"), field(b, "b"), field(H, "H")
    (all(>=(0), s0) && all(>=(0), q) && all(>=(0), bb) && all(>=(0), h)) ||
        throw(ArgumentError("sigma0, Q, b and H must be non-negative"))
    return VoxelPlasticMaterials(el.kappa, el.mu, s0, q, bb, h, el.grid)
end

gridsize(m::MaterialDistribution) = m.grid.size

"""
Isotropic reference medium (kappa0, mu0), from the elastic constants. For isotropic constituents, each
modulus is the midpoint of its extreme values (as in Moulinec & Suquet / Kairotop).
With anisotropic phases, the midpoint of the extreme Kelvin moduli (eigenvalues
of the Kelvin stiffness matrix) is used for both 3*kappa0 and 2*mu0.
"""
function reference_medium(m::PhaseMaterials)
    if all(x -> x isa Isotropic, m.materials)
        ks = [x.kappa for x in m.materials]
        ms = [x.mu for x in m.materials]
        return (minimum(ks) + maximum(ks)) / 2, (minimum(ms) + maximum(ms)) / 2
    end
    ev = reduce(vcat, [eigvals(Symmetric(stiffness(x))) for x in m.materials])
    l0 = (minimum(ev) + maximum(ev)) / 2
    return l0 / 3, l0 / 2
end

function reference_medium(m::Union{VoxelMaterials,VoxelPlasticMaterials})
    kmin, kmax = extrema(m.kappa)
    mmin, mmax = extrema(m.mu)
    return (kmin + kmax) / 2, (mmin + mmax) / 2
end
