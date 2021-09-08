function data = scale_data(data)
    %data = (data-mean(data))/std(data);
    data = data/range(data);
end