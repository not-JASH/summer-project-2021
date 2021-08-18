load("../delta_optimization/samples.mat");

%% Process Variables

nSamples = 1e4;
WindowSize = 360;
predLen = 5;

%% 

xData = cell(nSamples,1);
yData = cell(nSamples,1);

sampleLocs = randi(size(samples,1),[nSamples 1]);

getWorkers(6);

parfor i = 1:nSamples
    xo = randi(size(samples{sampleLocs(i)}.data,1)-WindowSize-1,1);
    xData{i} = samples{sampleLocs(i)}.zeromean(xo:xo+WindowSize-1);
    %xo = xo + predLen;
    yData{i} = samples{sampleLocs(i)}.bin(xo:xo+WindowSize-1);
    
    xData{i} = scaleData(xData{i});
end

% hotfix: input and output sequences must have dimensions C by S where C is
% the number of channels. (for trainNetwork)
for i = 1:nSamples
    xData{i} = xData{i}';
    yData{i} = yData{i}';
end


function data = scaleData(data)
    data = data-min(data);
    data = data/max(data);
    data = 2*data - 1;
end