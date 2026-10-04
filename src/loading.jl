export MacroLoading, StrainLoading, StressLoading, MixedLoading

"""
    MacroLoading

Macroscopic loading: for each Kelvin component, either the mean
strain or the mean stress is imposed. Build it with [`StrainLoading`](@ref),
[`StressLoading`](@ref) or [`MixedLoading`](@ref).
"""
struct MacroLoading
    value::Vector{Float64}            # Kelvin components
    stress_controlled::NTuple{6,Bool}
end

"""
    StrainLoading(E)

Impose the macroscopic strain `E`: a Kelvin 6-vector
`(E11, E22, E33, √2 E23, √2 E13, √2 E12)` or a symmetric 3x3 matrix.
"""
StrainLoading(E) = MacroLoading(as_kelvin(E), ntuple(_ -> false, 6))

"""
    StressLoading(S)

Impose the macroscopic stress `S`: a Kelvin 6-vector or a symmetric 3x3 matrix.
"""
StressLoading(S) = MacroLoading(as_kelvin(S), ntuple(_ -> true, 6))

"""
    MixedLoading(values, stress_controlled)

Mixed loading on Kelvin components: `values[k]` is the imposed mean stress
if `stress_controlled[k]`, the imposed mean strain otherwise. E.g. uniaxial stress along 1:
`MixedLoading([0.01,0,0,0,0,0], (false,true,true,true,true,true))`.
"""
MixedLoading(values, stress_controlled) =
    MacroLoading(as_kelvin(values), Tuple(Bool.(collect(stress_controlled))))
