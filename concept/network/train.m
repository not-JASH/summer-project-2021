addpath("../delta_optimization","./models");

getTrainingSamples;

model = [...
    sequenceInputLayer(1)
    
    bilstmLayer(128)
    dropoutLayer(0.2)
    reluLayer
    fullyConnectedLayer(size(xData{1},2))
    
    bilstmLayer(128)
    dropoutLayer(0.2)
    reluLayer
    
    bilstmLayer(128)
    dropoutLayer(0.2)
    reluLayer
    
    fullyConnectedLayer(1)
    regressionLayer
    ];
    
    
options = trainingOptions('adam',...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'Epsilon',1e-8,...
    'InitialLearnRate',5e-3,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',15,...
    'L2Regularization',0.0001,...
    'GradientThresholdMethod','l2norm',...
    'MaxEpochs',100,...
    'MiniBatchSize',128,...
    'Verbose',true,...
    'VerboseFrequency',100,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','gpu',...
    'Plots','none');

model = trainNetwork(xData,yData,model,options);