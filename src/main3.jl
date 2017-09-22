include("gumbel_softmax.jl")

using ReverseDiff

#= and(x, y) = Int.((x + y) .== 2) =#
#= or(x, y) = Int.((x + y) .>= 1) =#
#= xor(x, y) = Int.((x .* y) .== 0) =#
and(x, y) = x+y 
or(x, y) = x-y 
xor(x, y) = x*y 

gt(X) = and(xor(X[1],X[5]), or(X[2],X[4])) #(1,3,2,4,8,5,7)

const N = 6
X = hcat([parse.(Int,vcat(bin(i,N)...)) for i=0:2^N-1]...)' #all combos of bits up to N

function f1(ps1, ps2, ps3, ps4, ps5, ps6, ps7, x)
    T = 1.0
    logits = log.(ps1)
    w = gumbel_softmax(logits, T, true)
    return w' * [and(f2(ps2,ps4,ps5,x),f3(ps3,ps6,ps7,x)), or(f2(ps2,ps4,ps5,x),f3(ps3,ps6,ps7,x)), xor(f2(ps2,ps4,ps5,x),f3(ps3,ps6,ps7,x)), x[1], x[2], x[3], x[4], x[5], x[6]]
end
function f2(ps2, ps4, ps5, x)
    T = 1.0
    logits = log.(ps2)
    w = gumbel_softmax(logits, T, true)
    return w' * [and(f4(ps4,x),f5(ps5,x)), or(f4(ps4,x),f5(ps5,x)), xor(f4(ps4,x),f5(ps5,x)), x[1], x[2], x[3], x[4], x[5], x[6]]
end
function f3(ps3, ps6, ps7, x)
    T = 1.0
    logits = log.(ps3)
    w = gumbel_softmax(logits, T, true)
    return w' * [and(f6(ps6,x),f7(ps7,x)), or(f6(ps6,x),f7(ps7,x)), xor(f6(ps6,x),f7(ps7,x)), x[1], x[2], x[3], x[4], x[5], x[6]]
end
function f4(ps4, x)
    T = 1.0
    logits = log.(ps4)
    w = gumbel_softmax(logits, T, true)
    return w' * [x[1], x[2], x[3], x[4], x[5], x[6]]
end
function f5(ps5, x)
    T = 1.0
    logits = log.(ps5)
    w = gumbel_softmax(logits, T, true)
    return w' * [x[1], x[2], x[3], x[4], x[5], x[6]]
end
function f6(ps6, x)
    T = 1.0
    logits = log.(ps6)
    w = gumbel_softmax(logits, T, true)
    return w' * [x[1], x[2], x[3], x[4], x[5], x[6]]
end
function f7(ps7, x)
    T = 1.0
    logits = log.(ps7)
    w = gumbel_softmax(logits, T, true)
    return w' * [x[1], x[2], x[3], x[4], x[5], x[6]]
end
function objective(ps1, ps2, ps3, ps4, ps5, ps6, ps7, x, y_target)
    abs2.(y_target - f1(ps1, ps2, ps3, ps4, ps5, ps6, ps7, x))
end

Y = [gt(X[i,:]) for i=1:size(X,1)]
ps1 = ones(9) ./ 9
ps2 = deepcopy(ps1) 
ps3 = deepcopy(ps1) 
ps4 = ones(6) ./ 6 
ps5 = deepcopy(ps4) 
ps6 = deepcopy(ps4) 
ps7 = deepcopy(ps4) 
#= j = 20 =#
#= f1(ps1, ps2, ps3, ps4, ps5, ps6, ps7, X[j,:]) =#
#= g = ReverseDiff.gradient(objective, (ps1, ps2, ps3, ps4, ps5, ps6, ps7, X[j,:], [Y[j]])) =#

function learn(α=0.1, N=5)
    Y = [gt(X[i,:]) for i=1:size(X,1)]
    ps1 = ones(9) ./ 9
    ps2 = deepcopy(ps1) 
    ps3 = deepcopy(ps1) 
    ps4 = ones(6) ./ 6 
    ps5 = deepcopy(ps4) 
    ps6 = deepcopy(ps4) 
    ps7 = deepcopy(ps4) 

    for i = 1:N
        println("i=$i")
        @show ps1
        @show ps2
        @show ps3
        @show ps4
        @show ps5
        @show ps6
        @show ps7
        for j = 1:size(X,1)
            g1,g2,g3,g4,g5,g6,g7,_,_ = ReverseDiff.gradient(objective, (ps1, ps2, ps3, ps4, ps5, ps6, ps7, X[j,:], [Y[j]]))
            #= @show ps1 =#
            #= @show ps2 =#
            #= @show ps3 =#
            #= @show ps4 =#
            #= @show ps5 =#
            #= @show ps6 =#
            #= @show ps7 =#
            #= @show g1 =#
            #= @show g2 =#
            #= @show g3 =#
            #= @show g4 =#
            #= @show g5 =#
            #= @show g6 =#
            #= @show g7 =#
            ps1 = update(ps1, g1, α)
            ps2 = update(ps2, g2, α)
            ps3 = update(ps3, g3, α)
            ps4 = update(ps4, g4, α)
            ps5 = update(ps5, g5, α)
            ps6 = update(ps6, g6, α)
            ps7 = update(ps7, g7, α)
        end
    end
    return indmax(ps1), indmax(ps2), indmax(ps3), indmax(ps4), indmax(ps5), indmax(ps6), indmax(ps7)
end
function update(ps, g, α)
    if !any(isnan.(g))
        ps -= α * g
    end
    ps = max.(min.(ps, 1.0), 1e-12)
    normalize!(ps,1)
    ps
end

#=  =#
#= function test(n_seeds=50) =#
#=     result = zeros(length(OP), length(OP)) #truth x result =#
#=     for (i, gt) in enumerate(OP) =#
#=         for s=1:n_seeds =#
#=             srand(s) =#
#=             imax = learn(gt) =#
#=             result[i, imax] += 1 =#
#=         end =#
#=     end =#
#=     result =#
#= end =#

#Issues:
#1. y_hard: tell ReverseDiff to hard forward, soft back
#2. variable temperature: tell ReverseDiff that T is a constant 
#3. ReverseDiff throws and error when batch_size > 1 
