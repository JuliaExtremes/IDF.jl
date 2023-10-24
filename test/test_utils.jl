@testset "utils.jl" begin

    @testset "logistic()" begin

        x = rand()
        @test IDF.logistic(x) ≈ log(x/(1-x))

        @test_throws DomainError IDF.logistic(1, a=2)
        @test_throws DomainError IDF.logistic(3, b=2)

    end

    @testset "logistic_inverse()" begin

        x = rand()*20-10
        @test IDF.logistic_inverse(x) ≈ 1/(1+exp(-x))
        @test IDF.logistic_inverse(x, a=2, b=3) >= 2
        @test IDF.logistic_inverse(x, a=2, b=3) <= 3
        
    end

    @testset "to_french_name()" begin

        @test IDF.to_french_name(11) == "11 min"
        @test IDF.to_french_name(120) == "2 h"
        @test_throws InexactError IDF.to_french_name(61)

    end

    @testset "to_duration()" begin

        @test IDF.to_duration(IDF.to_french_name(11)) == 11
        @test IDF.to_duration("2 h") == IDF.to_duration("120 min") == 120
        @test_throws InexactError IDF.to_duration("2.5 min")

    end

    @testset "cvmcriterion()" begin

        distrib = Normal(0,1)
        x = IDF.Distributions.rand(distrib, 10)
        x̃ = sort(x)

        @test IDF.cvmcriterion(distrib,x) == 1/(12*10) + sum( ((2*i-1)/(2*10) - IDF.Distributions.cdf(distrib,x̃[i]) )^2 for i=1:10)

    end

    @testset "modifiedADcriterion()" begin

        distrib = Normal(0,1)
        x = IDF.Distributions.rand(distrib, 10)
        x̃ = sort(x)

        @test IDF.modifiedADcriterion(distrib,x) == 10/2 - 2*sum( IDF.Distributions.cdf(distrib,x̃[i]) for i in 1:10 ) - sum( ( 2-(2*i-1)/10 ) * log( 1 - IDF.Distributions.cdf(distrib,x̃[i]) ) for i=1:10)

    end

    @testset "approx_eigenvalues()" begin

        ρ(u,v) = u*v

        λs_1 = IDF.approx_eigenvalues(ρ, 10)
        @test length(λs_1) == 10
        @test λs_1[9] >= λs_1[10]

        λs_2 = IDF.approx_eigenvalues(ρ, 100)
        @test length(λs_2) == 100
        @test (λs_2[1] - 1/3)^2 <= 10^(-3)
        @test (λs_2[2] - 0.0)^2 <= 10^(-3)

    end

    @testset "ZolotarevDistrib" begin

        ρ(u,v) = minimum([u,v]) - u*v
        λs = IDF.approx_eigenvalues(ρ, 50)
        zolo_distrib = IDF.ZolotarevDistrib(λs)
        
        @testset "minimum" begin
            @test IDF.minimum(zolo_distrib) == 0.0
        end

        @testset "maximum" begin
            @test IDF.maximum(zolo_distrib) == +Inf
        end

        @testset "insupport" begin
            @test !IDF.insupport(zolo_distrib,-1)
            @test IDF.insupport(zolo_distrib,3)
        end

        @testset "cdf()" begin

            x = 0.001
            @test IDF.cdf(zolo_distrib, x) == 0.0

            x = 0.5
            λ₁ = λs[1]
            term1 = prod([ (1 - λs[i]/λ₁)^(-0.5) for i in 2:50])
            term2 = 1/IDF.gamma(0.5)
            term3 = ( x/(2*λ₁) )^(-0.5)
            term4 = exp( - (x/(2*λ₁)) )
            @test IDF.cdf(zolo_distrib, x) == 1 - term1 * term2 * term3 * term4

        end

        @testset "quantile()" begin

            p = 0.95
            @test IDF.cdf(zolo_distrib, IDF.quantile(zolo_distrib, p)) ≈ p

        end


    end

end

