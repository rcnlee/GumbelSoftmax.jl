#TODO:
#handle batches
#gradient propagation line

softmax(x) = exp.(x) ./ sum(exp.(x), ndims(x)) 

"""
Sample from Gumbel(0,1)
"""
function sample_gumbel(shape, eps::Float64=1e-20)
    u = rand(shape)
    return -log.(-log.(u + eps) + eps)
end

"""
Draw a sample from the Gumbel-softmax distribution
"""
function gumbel_softmax_sample(logits, temperature::Float64)
    @assert temperature > 0.0
    y = logits + sample_gumbel(size(logits))
    return softmax(y ./ temperature)
end
using ReverseDiff
"""
Sample from the Gumbel-Softmax distribution and optionally discretize.
Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if true, take argmax, but differential w.r.t. soft sample y
Returns:
    [batch_size, n_class] sample from the Gumbel-softmax distribution.
    If hard=true, then the returned sample will be one-hot, otherwise it
    will be a probability distribution that sums to 1 across classes.
"""
function gumbel_softmax(logits, temperature::Float64, hard::Bool=false)
    y = gumbel_softmax_sample(logits, temperature) 
    if hard
        imax = indmax(y)
        y_hard = [i == imax ? 1.0 : 0.0 for i=1:length(y)]
        #= m = maximum(y) =#
        #= y_hard = [yy == m ? 1.0 : 0.0 for yy in y] =#
        #y_hard = zeros(Float64, length(logits))
        #y_hard[imax] += 1.0
        #= y_hard = zeros(Int, size(y)) =#
        #= for i = 1:length(imax) =#
        #=     y_hard[i, imax[i]] = 1 =#
        #= end =#
        #y_hard = [ j==imax[i] ? 1 : 0 for i=1:size(y,1), j=1:size(y,2) ]
        #return stop_gradient(y - y) + y
        return stop_gradient(y_hard - y) + y
    end
    return y
end

function indmaxcols(X)
    return [indmax(X[i,:]) for i=1:size(X,1)] 
end

using ReverseDiff: TrackedReal, TrackedArray, ForwardOptimize
using ForwardDiff: Dual, value
stop_gradient(x) = x 
@inline function stop_gradient{N}(n::Dual{N})
    Dual(value(n), 0.0)
end
stop_gradient(t::TrackedArray) = map(ForwardOptimize(stop_gradient), t)
stop_gradient{T<:TrackedReal}(t::Array{T}) = map(ForwardOptimize(stop_gradient), t)


#test
#ReverseDiff.gradient(x->sum(gumbel_softmax(x,0.3,true)),(logits,))
#ReverseDiff.gradient(x->sum(stop_gradient(ones(length(logits))-x)+x), (logits,))
