function [xData,yData,yRef,y] = subsample(sample,no_samples,window_size,prediction_length,bins)
%function [xData,yData,yBin,yRef,y] = subsample(sample,no_samples,window_size,prediction_length,bins)

    xData = cell(no_samples,1);yData = cell(no_samples,1);
    yBin = cell(no_samples,1);yRef = cell(no_samples,1);
    y = cell(no_samples,1);
    
    for i = 1:no_samples
        [xData{i},yData{i},yRef{i},y{i}] = get_subsample(sample);
        %[xData{i},yData{i},yBin{i},yRef{i},y{i}] = get_subsample(sample);
    end

    function [xData,yData,yRef,y] = get_subsample(sample)
%     function [xData,yData,yBin,yRef,y] = get_subsample(sample)
        
        xo = randi(length(sample.data)-1-window_size-prediction_length,1);
%         xData = scale_data(sample.zeromean(xo:xo+window_size-1))';
        xData = cat_delta(sample.zeromean(xo:xo+window_size-1),bins)';
        
        xo = xo + prediction_length;
%         yData = sample.bin(xo:xo+window_size-1)';
        yRef = cat_delta(sample.zeromean(xo:xo+window_size-1)',bins);        
%         yBin = sample.bin(xo:xo+window_size-1)';
        y = sample.data(xo:xo+window_size-1);
        yData = zeros(length(bins),window_size);
        
        for j = 1:window_size
            yData(yRef(j),j) = 1;
        end                
    end
end