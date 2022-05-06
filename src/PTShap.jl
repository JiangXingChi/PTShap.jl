module PTShap

using DataFrames,StatsBase,ShapML,MLJ,Random

include("sampling.jl")
include("permutation.jl")
include("shaptable.jl")

export allcolsample,colsample,
       mystatistic,feature_permutation_test,
       predict_function,feature_shappley_value,shap_sample
       
end 
