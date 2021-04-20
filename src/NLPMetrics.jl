# NLP Metrics
module NLPMetrics

export bleu_score, rouge, rouge_l_summary_level

include("rouge.jl")
include("bleu.jl")

end
