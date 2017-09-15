include("gumbel_softmax.jl")

using ReverseDiff

and(x, y) = Int.((x + y) .== 2)
or(x, y) = Int.((x + y) .>= 1)
xor(x, y) = Int.((x .* y) .== 0)

const OP = [and, or, xor] 

X = hcat([[x1,x2] for x1 in (0,1), x2 in (0,1)]...)'

function f(ps, x1, x2)
    T = 0.3
    logits = log.(ps)
    w = gumbel_softmax(logits, T, true)
    return w' * [op(x1,x2) for op in OP]
end

function objective(ps, x1, x2, y_target)
    abs2.(y_target - f(ps, x1, x2))
end

function learn(gt::Function=xor, α=0.1, N=50)
    ps = ones(length(OP)) ./ length(OP)
    Y = [gt(X[i,1], X[i,2]) for i=1:size(X,1)] 
    for i = 1:N
        @show ps
        for j = 1:4
            g, _, _, _ = ReverseDiff.gradient(objective, (ps, [X[j,1]], [X[j,2]], [Y[j]]))
            #@show g
            if !any(isnan.(g))
                ps -= α * g
            end
            ps = max.(min.(ps, 1.0), 1e-12)
            normalize!(ps,1)
        end
    end
    return indmax(ps)
end

function test(n_seeds=50)
    result = zeros(length(OP), length(OP)) #truth x result
    for (i, gt) in enumerate(OP)
        for s=1:n_seeds
            srand(s)
            imax = learn(gt)
            result[i, imax] += 1
        end
    end
    result
end

#Issues:
#1. y_hard: tell ReverseDiff to hard forward, soft back
#2. variable temperature: tell ReverseDiff that T is a constant 
#3. ReverseDiff throws and error when batch_size > 1 
