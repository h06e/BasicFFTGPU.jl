export VTKImage, read_vtk, write_vtk, load_phase_materials, load_voxel_materials

#! VTK I/O uses the standard XML ImageData format (.vti): voxels are the
#! *cells* of the image, voxel values are stored as CellData. Files written
#! by WriteVTK.jl, ParaView, PyVista, ... are read via ReadVTK.jl.

"""
    VTKImage

Content of a `.vti` file: named voxel fields and the voxel grid geometry.
"""
struct VTKImage
    fields::Dict{String,Array}
    grid::VoxelGrid
end

Base.getindex(img::VTKImage, name::AbstractString) = img.fields[name]
Base.keys(img::VTKImage) = keys(img.fields)

"""
    read_vtk(path) -> VTKImage

Read a VTK XML ImageData file (`.vti`). Cell data are the voxel values. If the
file only holds point data, each point is treated as a voxel (grid size =
number of points per direction). Scalar fields come back as `nx x ny x nz`
arrays, `n`-component fields as `n x nx x ny x nz` arrays.
"""
function read_vtk(path::AbstractString)
    vtk = VTKFile(path)
    vtk.file_type == "ImageData" ||
        throw(ArgumentError("$path: expected a VTK ImageData (.vti) file, got $(vtk.file_type)"))

    data, cell_data = try
        get_cell_data(vtk), true
    catch
        nothing, false
    end
    if data === nothing || isempty(data.names)
        data, cell_data = get_point_data(vtk), false
    end

    fields = Dict{String,Array}(name => get_data_reshaped(arr; cell_data=cell_data) for (name, arr) in data)
    isempty(fields) && throw(ArgumentError("$path: no cell or point data found"))

    first_field = first(values(fields))
    sz = size(first_field)[end-2:end]
    spacing = pad3(get_spacing(vtk), 1.0)
    origin = pad3(get_origin(vtk), 0.0)
    # point data: point i sits at the voxel center -> lower corner is half a voxel before
    cell_data || (origin = origin .- spacing ./ 2)
    return VTKImage(fields, VoxelGrid(sz; spacing=spacing, origin=origin))
end

pad3(v, fill_value) = ntuple(k -> k <= length(v) ? Float64(v[k]) : fill_value, 3)

function scalar_field(img::VTKImage, name, path)
    haskey(img.fields, name) ||
        throw(ArgumentError("$path: no field \"$name\" (available: $(join(keys(img.fields), ", ")))"))
    f = img.fields[name]
    ndims(f) == 3 || throw(ArgumentError("$path: field \"$name\" is not a scalar field"))
    return f
end

"""
    load_phase_materials(path, materials; field=nothing) -> PhaseMaterials

Read a phase map from a `.vti` file and attach a material list to it.
`materials` is a `Dict` (or a vector of pairs) `label => material`, or a
vector of materials for labels `1:n`. `field` selects the integer field; by
default the file must contain a single field.
"""
function load_phase_materials(path::AbstractString, materials; field=nothing)
    img = read_vtk(path)
    if field === nothing
        length(img.fields) == 1 ||
            throw(ArgumentError("$path holds several fields ($(join(keys(img.fields), ", "))), choose one with `field=`"))
        field = first(keys(img.fields))
    end
    raw = scalar_field(img, field, path)
    all(x -> isinteger(x), raw) || throw(ArgumentError("$path: field \"$field\" is not an integer phase map"))
    phases = Int.(raw)
    mats = materials isa AbstractVector{<:Pair} ? Dict(materials...) : materials
    return PhaseMaterials(phases, mats; grid=img.grid)
end

"""
    load_voxel_materials(path; kappa=, mu=)            -> VoxelMaterials
    load_voxel_materials(path; E=, nu=)
    load_voxel_materials(path; lambda=, mu=)

Read voxel-wise isotropic elastic constants from a `.vti` file. Each keyword
gives the name of the field in `path` holding that constant, e.g.
`load_voxel_materials("micro.vti"; E="young", nu="poisson")`.
"""
function load_voxel_materials(path::AbstractString; kappa=nothing, mu=nothing, E=nothing, nu=nothing, lambda=nothing)
    img = read_vtk(path)
    rd(name) = name === nothing ? nothing : Float64.(scalar_field(img, name, path))
    return VoxelMaterials(; kappa=rd(kappa), mu=rd(mu), E=rd(E), nu=rd(nu), lambda=rd(lambda), grid=img.grid)
end

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Writing

"""
    write_vtk(path, fields::AbstractDict, grid::VoxelGrid)

Write voxel fields as cell data of a VTK ImageData file (`.vti` is appended by
WriteVTK). Values are `nx x ny x nz` arrays (scalars) or `n x nx x ny x nz`
arrays (`n`-component). Returns the list of written files.
"""
function write_vtk(path::AbstractString, fields::AbstractDict, grid::VoxelGrid)
    nx, ny, nz = grid.size
    vtk_grid(path, nx + 1, ny + 1, nz + 1; origin=grid.origin, spacing=grid.spacing) do vtk
        for (name, f) in fields
            vtk[String(name), VTKCellData()] = f
        end
    end
end

# ParaView reads 6-component arrays as symmetric tensors of plain tensor
# components ordered XX YY ZZ XY YZ XZ: Kelvin component, and its scale.
const PARAVIEW_FROM_KELVIN = ((1, 1.0), (2, 1.0), (3, 1.0), (6, 1 / sqrt(2.0)), (4, 1 / sqrt(2.0)), (5, 1 / sqrt(2.0)))

"Kelvin field (nx,ny,nz,6) -> ParaView symmetric tensor array (6,nx,ny,nz)."
function paraview_tensor(f::AbstractArray{T,4}) where {T<:Real}
    out = Array{T}(undef, 6, size(f, 1), size(f, 2), size(f, 3))
    for (c, (k, scale)) in enumerate(PARAVIEW_FROM_KELVIN)
        out[c, :, :, :] .= f[:, :, :, k] .* T(scale)
    end
    return out
end
