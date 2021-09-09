addpath("classes","functions","models");
clear_env();
%deepNetworkDesigner
%scale data with global mean and standard deviation?
%since mean->0, just sdev? -> range, sdev results in values >> 1 || << -1
%calibrate trader with trained model and non-training data


window_size = 90;
rate = 30;
no_samples = 3e4;
no_sets = 10;
prediction_length = 0;
time_before = 7*24*60;
time_after = 1*24*60;

batch_size = 128;

[xVal,yVal] = deal({},{});

options = trainingOptions('adam',...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'Epsilon',1e-8,...
    'InitialLearnRate',0.001,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',5,... 
    'L2Regularization',0.0001,...
    'GradientThresholdMethod','l2norm',...
    'GradientThreshold',inf,...
    'MaxEpochs',20,...
    'MiniBatchSize',batch_size,...
    'Verbose',true,...
    'VerboseFrequency',ceil(no_samples/batch_size),...
    'ValidationData',{xVal,yVal},...
    'ValidationFrequency',ceil(no_samples/batch_size),...
    'ValidationPatience',Inf,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','gpu',...
    'plots','none');

models = cell(no_sets,1);
results = zeros(no_sets,4);
lists = cell(no_sets,1);

for i = 1:no_sets
    %options.InitialLearnRate = 0.0001;
    %options.MaxEpochs = 5;
    [train_samples,eval_samples] = get_samples("BTCUSDT.txt",1,rate,time_before,time_after);
    
    try 
        models{i} = layers_1;
    catch 
        models{i} = lgraph_1;
    end
    
    [xData,yData] = subsample(train_samples{1},no_samples,window_size,prediction_length);
    [xVal,yVal] = subsample(eval_samples{1},0.2*no_samples,window_size,prediction_length);
    
    for j = 1:length(yData)
        yData{j} = yData{j}(end);
    end
    for j = 1:length(yVal)
        yVal{j} = yVal{j}(end);
    end
    [yData,yVal] = deal(...
        categorical(cell2mat(yData)),categorical(cell2mat(yVal)));
    
%     yData = to_categorical(yData);
%     yVal = to_categorical(yVal);
    
    options.ValidationData = {xVal,yVal};
    models{i} = trainNetwork(xData,yData,models{i},options);
    
    %options.InitialLearnRate = 10*options.InitialLearnRate;
    %options.MaxEpochs = 30;
    %model = trainNetwork(xData,yData,layerGraph(model),options)

    [results(i,1),results(i,2),results(i,3),results(i,4),lists{i}] = ...
        evaluate_model(eval_samples{1},models{i},window_size,prediction_length,false,false,0); 
    %evaluate_model(train_samples{1},models{i},window_size,prediction_length,false,false,0); 
end





function data = to_categorical(data)
    for i = 1:length(data)
        data{i} = categorical(data{i});
    end
end