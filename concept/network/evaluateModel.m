function [Error,roundedError] = evaluateModel(model,xData,yData,nSamples)
    Error = zeros(nSamples,1);
    roundedError = Error;
    
    i = randperm(size(xData,1));
    xData = xData(i);
    yData = yData(i);
    
    for i = 1:nSamples
        progressbar(i/nSamples);
        sample = predict(model,xData{i});
        Error(i) = mean((yData{i}-sample)/numel(sample));
        roundedError(i) = mean((yData{i}-round(sample))/numel(sample));        
    end
    
    fprintf("average error : %.6f\n",mean(Error));
    fprintf("average rounded error : %.6f\n",mean(roundedError));
end