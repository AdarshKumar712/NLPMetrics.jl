"""
    crossentropy(ŷ, y;  dims=dims, ϵ=epseltype(ŷ))

Returns the crossentropy loss for `ŷ` and `y` evaluated as,

    mean(.-sum(y.* log.(ŷ .+ ϵ); dims=dims))

Refer: https://github.com/FluxML/Flux.jl/blob/master/src/losses/functions.jl
"""
function crossentropy(ŷ, y; dims=1, ϵ=eps(float(eltype(ŷ))))
    mean(.-sum(y.* log.(ŷ .+ ϵ); dims=dims))
end

"""
    logitcrossentropy(ŷ, y; dims = 1)
    
Returns the logitcrossentropy loss for `ŷ` and `y` evaluated as,

    mean(.-sum(y .* logsoftmax(ŷ; dims = dims); dims = dims))

"""
function logitcrossentropy(ŷ, y; dims = 1)
    mean(.-sum(y .* logsoftmax(ŷ; dims = dims); dims = dims))
end

"""
    perplexity(y_pred, y_target; onehot=true, labels=nothing, loss=logitcrossentropy)

Returns the perplexity score based on the `loss` function specified.   Evaluated as,
    
    exp(loss(y_pred, y_target))

Please note that the function expects the `y_target` to be one hotencoded by default with `onehot=true`. 

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> y_ = [ 0.07; 0.60 ]
2-element Vector{Float64}:
 0.07
 0.6

julia> y = [0, 1]
2-element Vector{Int64}:
 0
 1

julia> perplexity(y_, y)
1.588604969678355

julia> y = [1]
1-element Vector{Int64}:
 1

julia> perplexity(y_, y; onehot=false, labels=0:1)
1.588604969678355

```
"""
function perplexity(y_pred, y_target; onehot=true, labels=nothing, loss=logitcrossentropy)
    if onehot==false
    	try
    	    y_target = onehot_encode(y_target, labels)
    	catch e
    	    print("`labels` not provided")
    	    throw(error()) 
    	end
    end
    exp(loss(y_pred, y_target))
end

# TODO: Add better example for Perplexity
