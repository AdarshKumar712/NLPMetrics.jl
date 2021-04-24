# NLP Metrics
module NLPMetrics

export bleu_score, rouge, rouge_l_summary_level
export onehot_encode, confusion_matrix, precision, recall, f_beta_score

include("rouge.jl")
include("bleu.jl")
include("classification.jl")

end
