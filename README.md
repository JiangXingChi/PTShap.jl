# PTShap.jl
* Permutation test and Shapley value.
* Install:
```
using Pkg
Pkg.add(PackageSpec(url="https://github.com/JiangXingChi/PTShap.jl"))
```


For example:
```
using DataFrames,StatsBase,ShapML,MLJ,Random
using RDatasets,CSV

iris=dataset("datasets","iris")
data=iris[:,Not(:Species)]
pretty(first(data,3))
outcome_name="PetalWidth"
y,x=unpack(data,==(Symbol(outcome_name)))
random_forest=@load RandomForestRegressor pkg=DecisionTree
model_arguments=random_forest(n_trees=100)

feature_sig=feature_permutation_test(model_arguments,x,y)

data_shap,data_feature=feature_shappley_value(model_arguments,x,y)

data_shap_sample=shap_sample(data_shap,x,y)

CSV.write("feature_sig.csv",feature_sig)
CSV.write("data_shap.csv",data_shap)
CSV.write("data_feature.csv",data_feature)
CSV.write("data_shap_sample.csv",data_shap_sample)
```
