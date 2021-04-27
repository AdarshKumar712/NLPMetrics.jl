"""
    confusion_matrix(y_pred, y_true)

Function to create a confusion_matrix for classification problems based on provided `y_pred` and `y_true`. Expects `y_true` and `y_pred`, to be onehot_enocded already.
"""
function confusion_matrix(y_pred, y_true)
    @assert size(y_pred) == size(y_true)
    label_count = size(y_true, 1)
    ŷ = onehot_encode(onecold(y_pred), 1:label_count)
    return ŷ * transpose(y_true) 
end

"""
    TFPN(y_pred, y_true)

Returns `Confusion Matrix` and `True Positive`, `True Negative`, `False Positive` and `False Negative` for each class based on `y_pred` and `y_true`. Expects `y_true`, to be onehot_enocded already.  
""" 
function TFPN(y_pred, y_true)
    @assert size(y_pred) == size(y_true)
    label_count = size(y_true, 1)
    TP = zeros(label_count) ; TN = zeros(label_count)
    FP = zeros(label_count) ; FN = zeros(label_count)
    ConfusionMatrix = confusion_matrix(y_pred, y_true)
    for c in 1 : label_count
        TP[c] = ConfusionMatrix[c,c]
        FP[c] = sum(ConfusionMatrix[:,c]) - TP[c]
        FN[c] = sum(ConfusionMatrix[c,:]) - TP[c]
        TN[c] = sum(ConfusionMatrix) - TP[c] - FP[c] - FN[c]
    end
    return ConfusionMatrix,TP, TN, FP, FN
end

"""
    precision(y_pred, y_true; avg_type="macro", sample_weights=nothing)

Computes the precision of the predictions with respect to the labels. 

# Arguments
 - `y_pred`: predicted values.
 - `y_true`: ground truth values on the basis of which predicted values are to be assessed. Expects it to be one-hot encoded already
 - `avg_type="macro"`: Type of average to be used while calculating precision of multiclass models. Can take values as `macro`, `micro` and `weighted`. Default set to `macro`.
 - `sample_weights`: Class weights to be provided when `avg_type` is set to `weighted`. Useful in case of imbalanced classes.
 
"""
function precision(y_pred, y_true; avg_type="macro", sample_weights=nothing)
    _,TP, TN, FP, FN = TFPN(y_pred, y_true)
    # Macro-averaged Precision
    if avg_type == "macro"
        return mean(TP ./ (TP .+ FP .+ eps(eltype(TP))))
    # Micro-averaged Precision
    elseif avg_type == "micro"   
        return mean(TP) / (mean(TP) + mean(FP))
    # Weighted-Averaged Precision
    elseif avg_type == "weighted"
        weights = []
        if sample_weights != nothing
            weights = sample_weights
        else
            weights = calc_instances(y_true) / size(y_true, 2)
        end
        return mean((TP ./ (TP .+ FP .+ eps(eltype(TP)))) .* weights)
    end
end

"""
    recall(y_pred, y_true; avg_type="macro", sample_weights=nothing)

Computes the recall of the predictions with respect to the labels.

# Arguments
 - `y_pred`: predicted values. 
 - `y_true`: ground truth values on the basis of which predicted values are to be assessed. Expects it to be one-hot encoded already.
 - `avg_type="macro"`: Type of average to be used while calculating precision of multiclass models. Can take values as `macro`, `micro` and `weighted`. Default set to `macro`.
 - `sample_weights`: Class weights to be provided when `avg_type` is set to `weighted`. Useful in case of imbalanced classes.

Aliases: `sensitivity` and `detection_rate`
"""
function recall(y_pred, y_true; avg_type="macro", sample_weights=nothing)
    _,TP, TN, FP, FN = TFPN(y_pred, y_true)
    # Macro-averaged Precision
    if avg_type == "macro"
        return mean(TP ./ (TP .+ FN .+ eps(eltype(TP))))    
    # Micro-averaged Precision
    elseif avg_type == "micro"   
        return mean(TP) / (mean(TP) + mean(FN))
    # Weighted-Averaged Precision
    else
        weights = []
        if sample_weights != nothing
            weights = sample_weights
        else
            weights = calc_instances(y_true) / size(y_true, 2)
        end
        return mean((TP ./ (TP .+ FN .+ eps(eltype(TP)))) .* weights)
    end
end
const sensitivity = recall
const detection_rate = recall

"""
    f_beta_score(y_pred, y_true; β=1, avg_type="macro", sample_weights=nothing)

Compute f-beta score. The F_beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.

# Arguments
 - `y_pred`: predicted values.
 - `y_true`: ground truth values on the basis of which predicted values are to be assessed. Expects it to be one-hot encoded already
 - `β=1`: the weight of precision in the combined score. If `β<1`, more weight given to `precision`, while `β>1` favors recall.
 - `avg_type="macro"`: Type of average to be used while calculating precision of multiclass models. Can take values as `macro`, `micro` and `weighted`. Default set to `macro`.
 - `sample_weights`: Class weights to be provided when `avg_type` is set to `weighted`. Useful in case of imbalanced classes.

"""
function f_beta_score(y_pred, y_true; β=1, avg_type="macro", sample_weights=nothing)
    recall_ = recall(y_pred, y_true, avg_type=avg_type, sample_weights=sample_weights)
    precision_ = precision(y_pred, y_true, avg_type=avg_type, sample_weights=sample_weights)
    return (1 + β^2) * precision_ * recall_ / (precision_ + (β^2) * recall_ + eps(eltype(y_pred)))
end

# TODO: add autocheck for one-hot encode of y_true and perform one-hot encoding if not already.
# TODO: add example of usage for each function

