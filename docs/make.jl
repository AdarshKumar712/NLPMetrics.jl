using NLPMetrics
using Documenter

DocMeta.setdocmeta!(NLPMetrics, :DocTestSetup, :(using NLPMetrics); recursive=true)

makedocs(;
    modules=[NLPMetrics],
    authors="Adarsh Kumar",
    repo="https://github.com/adarshkumar712/NLPMetrics.jl/blob/{commit}{path}#{line}",
    sitename="NLPMetrics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://adarshkumar712.github.io/NLPMetrics.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/AdarshKumar712/NLPMetrics.jl",
    devbranch="main",
)
