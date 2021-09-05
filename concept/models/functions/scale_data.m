function data = scaleData(data)
    data = (data-mean(data))/std(data);
end