using DataFrames,StatsBase,ShapML,MLJ,Random

function predict_function(model,x)
  data_pred=DataFrames.DataFrame(y_pred=MLJ.predict(model,x))
  return data_pred
end

"""
# Arguments
* `model_arguments`:Model parameters of MLJ.
* `x`:Feature data.
* `y`:Fitting value.
* `monte_carlo_samples`:Number of Monte Carlo samples.
* `seed_number`:Random seed.
# Return
* `data_shap`:Shapley value of the features of each sample.
* `data_feature`:The absolute average of Shapley value of each feature.
"""
function feature_shappley_value(model_arguments,x,y;monte_carlo_samples=100,seed_number=1)
  model=MLJ.machine(model_arguments,x,y)
  MLJ.fit!(model)
  data_shap=ShapML.shap(explain=copy(x),model=model,predict_function=predict_function,sample_size=monte_carlo_samples,seed=seed_number)
  data_shap_group=DataFrames.groupby(data_shap,:feature_name)
  shap_mean_effect=DataFrames.combine(data_shap_group,:shap_effect => (x->y=mean(abs.(x))) => :mean_effect)
  data_feature=DataFrames.sort(shap_mean_effect,order(:mean_effect,rev=true))
  return(data_shap,data_feature)
end

"""
# Arguments
* `data_shap`:Shapley value of the features of each sample.
* `x`:Feature data.
* `y`:Fitting value.
# Return
* `data_shap_sample`:Shapley value of each sample.
"""
function shap_sample(data_shap,x,y)
  data_shap_sample_raw=copy(data_shap[:,[:index,:feature_name,:shap_effect]])
  data_feature_n=DataFrames.names(x)
  imax=size(data_feature_n,1)
  indexmax=size(x,1)
  data_shap_sample=DataFrames.DataFrame(index=1:indexmax)
  for i in 1:imax
    data_temp=data_shap_sample_raw[data_shap_sample_raw.feature_name .== data_feature_n[i],:]
    DataFrames.sort!(data_temp,[:index],rev=(false))
    data_shap_sample_add1=DataFrames.DataFrame(shap_effect=data_temp[:,:shap_effect])
    DataFrames.rename!(data_shap_sample_add1,[Symbol(data_feature_n[i])])
    data_shap_sample=DataFrames.hcat(data_shap_sample,data_shap_sample_add1)
  end
  data_shap_sample_feature=DataFrames.copy(data_shap_sample[:,2:imax+1])
  data_shap_sample_sum=DataFrames.select(data_shap_sample_feature,All() => +)
  DataFrames.rename!(data_shap_sample_sum,[:shap_effect_sample_sum])
  data_shap_sample=DataFrames.hcat(data_shap_sample,data_shap_sample_sum)
  data_shap_sample_y=DataFrames.DataFrame(y_real=y)
  data_shap_sample=DataFrames.hcat(data_shap_sample,data_shap_sample_y)
  return(data_shap_sample)
end