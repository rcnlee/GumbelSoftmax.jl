include("gumbel_softmax.jl")

using ReverseDiff

const OP = [+, -, *, /] 

X = collect(1:5) 
gt1(x) = x + 1 #(1,1,2)
gt2(x) = x - 2  #(2,1,3)
gt3(x) = x * 3  #(3,1,4)
gt4(x) = x / 3  #(4,1,4)

gt = gt1

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

using DataFrames, PGFPlots, TikzPictures
TikzPictures.standaloneWorkaround(true)
function learn(α=0.0002, N=5000; track=false,
               D=track ? DataFrame([Int, Int, Float64, Float64, Float64, Float64], [:iter, :node, :p1, :p2, :p3, :p4], 0) : 0)
    ps1 = ones(length(OP)) ./ length(OP)
    ps2 = ones(4) ./ 4
    ps3 = ones(4) ./ 4
    Y = [gt(x) for x in X] 
    for i = 1:N
        @show i
        @show ps1
        @show ps2
        @show ps3
        if track
            push!(D, vcat(i, 1, ps1))
            push!(D, vcat(i, 2, ps2))
            push!(D, vcat(i, 3, ps3))
        end
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
    return D, indmax(ps1), indmax(ps2), indmax(ps3)
end

function plot_me(D::DataFrame)
    g = GroupPlot(3, 1, groupStyle="horizontal sep=1.75cm, vertical sep=1.5cm")
    for i = 1:3
        ax = Axis([
                   Plots.Linear(D[D[:node].==i,:][:p1], legendentry="node$i p1"),
                   Plots.Linear(D[D[:node].==i,:][:p2], legendentry="node$i p2"),
                   Plots.Linear(D[D[:node].==i,:][:p3], legendentry="node$i p3"),
                   Plots.Linear(D[D[:node].==i,:][:p4], legendentry="node$i p4"),
                  ])
        push!(g, ax)
    end
    save("main2.pdf", g)
    save("main2.tex", g)
end
