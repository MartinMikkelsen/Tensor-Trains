using Test

include("../src/tt_interpolations.jl")

@testset "Node Generation Tests" begin
    @testset "Chebyshev Lobatto Nodes" begin
        nodes = chebyshev_lobatto_nodes(4)
        @test length(nodes) == 5
        @test nodes[1] == 1.0
        @test nodes[end] == 0.0
    end

    @testset "Equally Spaced Nodes" begin
        nodes = equally_spaced_nodes(4)
        @test length(nodes) == 5
        @test nodes[1] == 0.0
        @test nodes[end] == 1.0
    end

    @testset "Legendre Nodes" begin
        nodes = legendre_nodes(4)
        @test length(nodes) == 4
        @test all(nodes .>= 0.0) && all(nodes .<= 1.0)
    end

    @testset "Get Nodes" begin
        @test length(get_nodes(4, "chebyshev")) == 5
        @test length(get_nodes(4, "equally_spaced")) == 5
        @test length(get_nodes(4, "legendre")) == 4
        @test_throws ErrorException get_nodes(4, "unknown")
    end
end

@testset "Lagrange Basis Tests" begin
    nodes = chebyshev_lobatto_nodes(4)
    
    @testset "Lagrange Basis for Scalar x" begin
        x = 0.5
        for j in 0:4
            basis = lagrange_basis(nodes, x, j)
            @test isfinite(basis)
        end
    end
end