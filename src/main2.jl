include("gumbel_softmax.jl")

using ReverseDiff

const OP = [+, -, *, /] 

X = collect(1:5) 
gt(x) = x + 1 #(1,1,2)
#gt(x) = x - 2  #(2,1,3)
#gt(x) = x * 3  #(3,1,4)
#gt(x) = x / 3  #(4,1,4)

function f1(ps1, ps2, ps3, x)
    T = 0.3
    logits = log.(ps1)
    w = gumbel_softmax(logits, T, true)
    return w' * [op.(f2(ps2,x), f3(ps3,x)) for op in OP]
end
function f2(ps2, x)
    T = 0.3
    logits = log.(ps2)
    w = gumbel_softmax(logits, T, true)
    return w' * [x, 1, 2, 3]
end
function f3(ps3, x)
    T = 0.3
    logits = log.(ps3)
    w = gumbel_softmax(logits, T, true)
    return w' * [x, 1, 2, 3]
end

function objective(ps1, ps2, ps3, x, y_target)
    abs2.(y_target - f1(ps1, ps2, ps3, x))
end

function learn(α=0.0002, N=5000)
    ps1 = ones(length(OP)) ./ length(OP)
    ps2 = ones(4) ./ 4
    ps3 = ones(4) ./ 4
    Y = [gt(x) for x in X] 
    for i = 1:N
        @show i
        @show ps1
        @show ps2
        @show ps3
        for j = 1:length(Y)
            g1, g2, g3, _, _ = ReverseDiff.gradient(objective, (ps1, ps2, ps3, [X[j]], [Y[j]]))
            #@show g
            if !any(isnan.(g1))
                ps1 -= α * g1
            end
            if !any(isnan.(g2))
                ps2 -= α * g2
            end
            if !any(isnan.(g3))
                ps3 -= α * g3
            end
            ps1 = max.(min.(ps1, 1.0), 1e-12)
            normalize!(ps1,1)
            ps2 = max.(min.(ps2, 1.0), 1e-12)
            normalize!(ps2,1)
            ps3 = max.(min.(ps3, 1.0), 1e-12)
            normalize!(ps3,1)
        end
        if isapprox(maximum(ps1),1) && isapprox(maximum(ps2),1) && isapprox(maximum(ps3),1)
            break
        end
    end
    return indmax(ps1), indmax(ps2), indmax(ps3)
end

function test(n_seeds=5)
    result = Tuple[] 
    for s=1:n_seeds
        srand(s)
        imax = learn()
        push!(result, imax)
    end
    result
end

