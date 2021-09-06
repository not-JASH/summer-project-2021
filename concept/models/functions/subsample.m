function [xData,yData] = subsample(sample,no_samples,window_size,prediction_length)
    
    xData = cell(no_samples,1);yData = cell(no_samples,1);
    for i = 1:no_samples
        [xData{i},yData{i}] = get_subsample(sample);
    end

    function [xData,yData] = get_subsample(sample)
        xo = randi(length(sample.data)-1-window_size-prediction_length,1);
        xData = scale_data(sample.zeromean(xo:xo+window_size-1))';
        xo = xo + prediction_length;
        yData = sample.bin(xo:xo+window_size-1)';
    end
end