using DataFrames,StatsBase,ShapML,MLJ,Random

function mystatistic(model,K,R,M)
  evaluate_data=MLJ.evaluate!(model,resampling=CV(nfolds=K,shuffle=true),measure=M,repeats=R)
  mean_Mvalue=mean(mean(evaluate_data.per_fold))
  return(mean_Mvalue)
end

"""
# Arguments
* `model_arguments`:Model parameters of MLJ.
* `x`:Feature data.
* `y`:Fitting value.
* `K`:K-fold cross validation.
* `M`:Calculation method of loss function.
* `N`:Number of permutation test.
* `digits_number`:How many decimal places are reserved.
# Return
* `feature_sig`:The result of permutation test.
"""
function feature_permutation_test(model_arguments,x,y;K=10,R=10,M=rmse,N=99,digits_number=2)
  model=MLJ.machine(model_arguments,x,y)
  divide_statistic=mystatistic(model,K,R,M)
  a,b=size(x)
  feature_sig=DataFrames.DataFrame(Feature=String[],Pvalue=[])
  name_data=DataFrames.names(x)
  temp_all_count=fill(0,(N,1))
  for k in 1:N
    new_x=allcolsample(x)
    new_model=MLJ.machine(model_arguments,new_x,y)
    temp_statistic=mystatistic(new_model,K,R,M)
    if  temp_statistic < divide_statistic
      temp_all_count[j]=1
    end
  end
  model_p_value=(sum(temp_all_count)+1)/(N+1)
  model_p_value=round(model_p_value;digits=digits_number)
  DataFrames.push!(feature_sig,("Full Model",model_p_value))
  for i in 1:b
    temp_feature_count=fill(0,(N,1))
    for j in 1:N
      new_x=colsample(x,i)
      new_model=MLJ.machine(model_arguments,new_x,y)
      temp_statistic=mystatistic(new_model,K,R,M)
      if  temp_statistic < divide_statistic
        temp_feature_count[j]=1
      end
    end
    p_value=(sum(temp_feature_count)+1)/(N+1)
    p_value=round(p_value;digits=digits_number)
    DataFrames.push!(feature_sig,(name_data[i],p_value))
  end
  return(feature_sig)
end