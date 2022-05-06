using DataFrames,StatsBase,ShapML,MLJ,Random

function allcolsample(x)
  new_x=copy(x)
  a,b=size(new_x)
  for i in 1:b
    new_x[:,i]=StatsBase.sample(x[:,i],a,replace=false) 
  end
    return(new_x)
end

function colsample(x,i)
  new_x=copy(x)
  a,b=size(new_x)
  new_x[:,i]=StatsBase.sample(x[:,i],a,replace=false)
  return(new_x)
end
    