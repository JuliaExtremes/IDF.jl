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

end

