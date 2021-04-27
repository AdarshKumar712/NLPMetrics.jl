# NLP Metrics
module NLPMetrics

export bleu_score, rouge, rouge_l_summary_level
export onehot_encode, confusion_matrix, precision, recall, f_beta_score
export perplexity

using Statistics
using NNlib


include("rouge.jl")
include("bleu.jl")
include("classification.jl")
include("perplexity.jl")
include("utils.jl")

end
