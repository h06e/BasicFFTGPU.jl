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

    @testset "elastoplasticity (J2, Voce)" begin
        # uniform material under uniaxial stress: sigma11 = R(p), p = eps11 - sigma11 / E
        E, nu, s0, Q, b, H = 200.0, 0.3, 1.0, 0.5, 50.0, 2.0
        R(p) = s0 + H * p + Q * (1 - exp(-b * p))
        pm = VoxelPlasticMaterials(E=fill(E, 4, 4, 4), nu=nu, sigma0=s0, Q=Q, b=b, H=H)
        ws = Workspace(pm; precision=Float64)
        uniax(e) = MixedLoading([e, 0, 0, 0, 0, 0], (false, true, true, true, true, true))
        local r
        for e in range(0, 0.02; length=11)[2:end]
            r = solve!(ws, uniax(e); tol=1e-10, warm_start=true)
            commit!(ws)
        end
        σ11 = r.mean_stress[1]
        @test r.converged
        @test σ11 ≈ R(0.02 - σ11 / E) rtol = 1e-8
        @test all(cumulated_plastic_strain(ws) .≈ 0.02 - σ11 / E)
        @test plastic_strain(ws)[1, 1, 1, 2] ≈ -plastic_strain(ws)[1, 1, 1, 1] / 2   # isochoric
        # elastic unloading, then back to zero strain
        r2 = solve!(ws, uniax(0.019); tol=1e-10, warm_start=true)
        @test r2.mean_stress[1] ≈ σ11 - 0.001E rtol = 1e-8
        r3 = solve!(ws, uniax(0.0); tol=1e-10, warm_start=true)
        @test r3.converged && r3.mean_stress[1] < 0

        # very high yield stress: elastic solution of VoxelMaterials
        ph = sphere(16, 5)
        Ev = [p == 1 ? 50.0 : 1.0 for p in ph]
        load = StrainLoading([0.0, 0.01, 0, 0, 0, 0])
        ref = solve(VoxelMaterials(E=Ev, nu=0.3), load; precision=Float64, tol=1e-10)
        rel = solve(VoxelPlasticMaterials(E=Ev, nu=0.3, sigma0=1e6), load; precision=Float64, tol=1e-10)
        @test rel.mean_stress ≈ ref.mean_stress

        # soft plastic matrix around a stiff elastic sphere: converges, no commit -> same answer
        pm = VoxelPlasticMaterials(E=Ev, nu=0.3, sigma0=[p == 1 ? 1e6 : 0.005 for p in ph], Q=0.005, b=20.0)
        ws = Workspace(pm; precision=Float64)
        r1 = solve!(ws, StrainLoading([0.02, 0, 0, 0, 0, 0]); tol=1e-8)
        r2 = solve!(ws, StrainLoading([0.02, 0, 0, 0, 0, 0]); tol=1e-8, warm_start=true)
        @test r1.converged && r2.converged
        @test r1.mean_stress ≈ r2.mean_stress rtol = 1e-6
        p = cumulated_plastic_strain(ws)
        @test maximum(p[ph .== 1]) == 0 && maximum(p) > 0
        mktempdir() do dir
            write_vtk(joinpath(dir, "plastic"), r2; material=pm, workspace=ws, fields=Dict("x" => p))
            img = read_vtk(joinpath(dir, "plastic.vti"))
            @test img["plastic_strain_12"] ≈ plastic_strain(ws)[:, :, :, 6] ./ sqrt(2)
            @test img["cumulated_plastic_strain"] ≈ p && img["x"] ≈ p
        end
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
            @test img["stress_11"] ≈ r.stress[:, :, :, 1]
            @test img["stress_23"] ≈ r.stress[:, :, :, 4] ./ sqrt(2)
        end
    end
end
