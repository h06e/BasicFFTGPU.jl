using BasicFFTGPU
using LinearAlgebra
using Test

const B = BasicFFTGPU

"Exact effective stiffness (Kelvin) of a laminate with layers normal to e1."
function laminate_stiffness(Cs, fracs)
    N = [1, 6, 5]; T = [2, 3, 4]       # components with continuous stress / continuous strain
    avg(f) = sum(fr * f(C) for (C, fr) in zip(Cs, fracs))
    CNN = inv(avg(C -> inv(C[N, N])))
    A = avg(C -> C[N, N] \ C[N, T])
    CTT = avg(C -> C[T, T] - C[T, N] * (C[N, N] \ C[N, T])) + avg(C -> C[T, N] / C[N, N]) * CNN * A
    Ce = zeros(6, 6)
    Ce[N, N] = CNN; Ce[N, T] = CNN * A; Ce[T, N] = (CNN * A)'; Ce[T, T] = CTT
    return Ce
end

m1 = Isotropic(E=1.0, nu=0.3)
m2 = Isotropic(E=50.0, nu=0.2)

sphere(n, r) = [(i - n / 2 - 0.5)^2 + (j - n / 2 - 0.5)^2 + (k - n / 2 - 0.5)^2 < r^2 ? 1 : 0
                for i in 1:n, j in 1:n, k in 1:n]

@testset "BasicFFTGPU" begin
    @testset "materials" begin
        m = Isotropic(E=210.0, nu=0.3)
        @test m.kappa ≈ 210 / (3 * 0.4)
        @test m.mu ≈ 210 / 2.6
        @test Isotropic(lambda=m.kappa - 2m.mu / 3, mu=m.mu).kappa ≈ m.kappa
        a = [1.0 0.4 0.5; 0.4 2.0 0.6; 0.5 0.6 3.0]
        @test kelvin(a) ≈ [1, 2, 3, 0.6sqrt(2), 0.5sqrt(2), 0.4sqrt(2)]
        @test kelvin_to_tensor(kelvin(a)) ≈ a
        C = isotropic_stiffness(m.kappa, m.mu)
        b = [0.1 0.2 0.0; 0.2 -0.3 0.4; 0.0 0.4 0.5]
        σ = 3m.kappa * tr(b) / 3 * I + 2m.mu * (b - tr(b) / 3 * I)
        @test C * kelvin(b) ≈ kelvin(σ)              # Kelvin: C : eps is a matrix product
        @test dot(kelvin(a), kelvin(b)) ≈ sum(a .* b)  # and a : b a dot product
        R = [cos(0.7) -sin(0.7) 0; sin(0.7) cos(0.7) 0; 0 0 1] * [1 0 0; 0 cos(0.3) -sin(0.3); 0 sin(0.3) cos(0.3)]
        Q = rotation_kelvin(R)
        @test Q' * Q ≈ I(6)
        @test Q * kelvin(a) ≈ kelvin(R * a * R')
        @test rotate_stiffness(C, R) ≈ C                     # isotropic is invariant
        Cc = cubic_stiffness(168.4, 121.4, 75.4)
        @test cubic_stiffness(10.0, 4.0, 3.0) ≈ isotropic_stiffness(6.0, 3.0)  # c11 - c12 = 2 c44
        @test eigvals(Symmetric(rotate_stiffness(Cc, R))) ≈ eigvals(Symmetric(Cc))
        @test StrainLoading(a).value ≈ kelvin(a)
        @test_throws ArgumentError Isotropic(E=1.0)
        @test_throws ArgumentError PhaseMaterials(zeros(Int, 2, 2, 2), Dict(1 => m))
    end

    @testset "laminate is exact ($g)" for g in (:continuous, :staggered)
        lam = ones(Int, 12, 8, 8)
        lam[1:4, :, :] .= 2
        C = effective_stiffness(PhaseMaterials(lam, [m1, m2]); green=g, precision=Float64,
                                tol=1e-10)
        Cex = laminate_stiffness([B.stiffness(m1), B.stiffness(m2)], [8 / 12, 4 / 12])
        @test C ≈ Cex rtol = 1e-8
    end

    mat = PhaseMaterials(sphere(16, 5), 0 => m1, 1 => m2)

    @testset "strain / stress / mixed loading consistency ($g)" for g in (:continuous, :staggered)
        C = effective_stiffness(mat; green=g, precision=Float64, tol=1e-9)
        S = [1.0, 0, 0, 0.3, 0, 0]
        r = solve(mat, StressLoading(S); green=g, precision=Float64, tol=1e-9)
        @test r.converged
        @test r.mean_stress ≈ S atol = 1e-8
        @test r.mean_strain ≈ C \ S rtol = 1e-6

        rm = solve(mat, MixedLoading([0.01, 0, 0, 0, 0, 0], (false, true, true, true, true, true));
                   green=g, precision=Float64, tol=1e-9)
        @test rm.mean_strain[1] ≈ 0.01
        @test norm(rm.mean_stress[2:6]) < 1e-8
    end

    @testset "accelerated and basic schemes agree" begin
        load = StrainLoading([0.01, 0, 0, 0, 0, 0.005])
        a = solve(mat, load; precision=Float64, tol=1e-9)
        b = solve(mat, load; precision=Float64, tol=1e-9, accelerate=false)
        @test a.mean_stress ≈ b.mean_stress rtol = 1e-6
        @test a.iterations < b.iterations
    end

    @testset "voxel-wise and anisotropic materials" begin
        ph = sphere(16, 5)
        vm = VoxelMaterials(E=[p == 1 ? 50.0 : 1.0 for p in ph], nu=[p == 1 ? 0.2 : 0.3 for p in ph])
        an = PhaseMaterials(ph, 0 => m1, 1 => Anisotropic(isotropic_stiffness(m2.kappa, m2.mu)))
        load = StrainLoading([0.0, 0.01, 0, 0, 0, 0])
        ref = solve(mat, load; precision=Float64, tol=1e-10)
        @test solve(vm, load; precision=Float64, tol=1e-10).mean_stress ≈ ref.mean_stress
        # anisotropic path uses another reference medium: equal up to solver tolerance
        @test solve(an, load; precision=Float64, tol=1e-10).mean_stress ≈ ref.mean_stress rtol = 1e-7
    end

    @testset "VTK round trip" begin
        mktempdir() do dir
            ph = sphere(8, 3)
            grid = VoxelGrid((8, 8, 8); spacing=(0.1, 0.1, 0.1))
            write_vtk(joinpath(dir, "micro"), Dict("phase" => Int32.(ph), "E" => 1.0 .+ ph, "nu" => fill(0.3, 8, 8, 8)), grid)
            m = load_phase_materials(joinpath(dir, "micro.vti"), Dict(0 => m1, 1 => m2); field="phase")
            @test B.phase_labels(m) == ph
            @test m.grid.spacing == (0.1, 0.1, 0.1)
            v = load_voxel_materials(joinpath(dir, "micro.vti"); E="E", nu="nu")
            @test size(v.kappa) == (8, 8, 8)

            r = solve(m, StrainLoading([0.01, 0, 0, 0, 0, 0]))
            write_vtk(joinpath(dir, "result"), r; material=m)
            img = read_vtk(joinpath(dir, "result.vti"))
            @test size(img["stress"]) == (6, 8, 8, 8)
            @test img["stress_kelvin"][4, :, :, :] ≈ r.stress[:, :, :, 4]
            @test img["stress"][5, :, :, :] ≈ r.stress[:, :, :, 4] ./ sqrt(2)  # ParaView YZ = sigma_23
            @test img["phase"] == ph
        end
    end
end
