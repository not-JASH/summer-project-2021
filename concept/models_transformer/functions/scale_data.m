function data = scale_data(data)
    %data = (data-mean(data))/std(data);
    %data = data/range(data);
    
    data = data - min(data);
    data = data/max(data);
    
    %outlier removal might help when a sudden jump occurs
    
end