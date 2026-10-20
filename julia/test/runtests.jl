using Test, MolLPGMPrep, Random
const P = MolLPGMPrep
@testset "Brownian bridge composition" begin
    config = MolecularPrepConfig(coordinate_process=P.TIME_CHANGED_BM_PROCESS)
    rng = MersenneTwister(43)
    for _ in 1:100
        s,t,v = sort(rand(rng,3))
        r,p,q = P._atom_bm_coefficients(s,t,config)
        r2,p2,q2 = P._atom_bm_coefficients(t,v,config)
        @test collect(P._atom_bm_coefficients(s,v,config)) ≈ [r*r2,p*r2+p2,q*r2*r2+q2]
        @test q >= 0
    end
    @test P._atom_bm_coefficients(.999,1.,config) == (0.,1.,0.)
    @test P._atom_bm_variance(.5,config) ≈ 14.8*(1-(1-.5^2.3)^3.1)*(1-.5^2.3)^3.1
end
