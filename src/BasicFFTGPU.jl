"""
    BasicFFTGPU

Basic FFT-based homogenization solver (Moulinec & Suquet) for linear elastic
voxel microstructures, running on CPU (`Array`, FFTW) or GPU (`CuArray`,
cuFFT) through KernelAbstractions kernels.

See `docs/CONVENTIONS.md` for the Kelvin notation used for all tensors.
"""
module BasicFFTGPU

using LinearAlgebra
using Printf
using AbstractFFTs
using FFTW
using KernelAbstractions
using ReadVTK
using WriteVTK

include("tensors.jl")
include("materials.jl")
include("loading.jl")
include("vtk.jl")
include("green.jl")
include("constitutive.jl")
include("workspace.jl")
include("solver.jl")

end # module BasicFFTGPU
