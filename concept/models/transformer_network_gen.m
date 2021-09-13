addpath("classes","functions","models");
clear_env();


%% define variables

datafile = "BTCUSDT.txt";

% data generation varaiables

window_size = 90;
rate = 30;
no_samples = 1e4;
no_sets = 10;
prediction_length = 0;
time_before = 7*24*60;
time_after = 1*24*60;


% trader variables

confidence_interval = 0.4;
heuristic_limit = 0;

% training variables

dropout_rate = 0.0;
batch_size = 32;
learn_rate = 1e-3;
max_epochs = 30;
valid_split = 0.2;

% transformer variables

input_channels = 1; 
target_size = 1; 
no_layers = 4;
d_model = 512;
no_heads = 8;
dff = 2048;


%% training loop

model = cell(no_sets,1);

for s = 1:no_sets
    
    %   initialize network & generate training/evaluation data
    
    net = transformer(...
        input_channels,target_size,no_layers,d_model,no_heads,dff,dropout_rate);
    
    [train_samples,eval_samples] = get_samples(datafile,1,rate,time_before,time_after);
    [xData,yData] = subsample(train_samples{1},no_samples,window_size+1,prediction_length);
    %[xVal,yVal] = subsample(eval_samples{1},valid_split*no_samples,window_size+1,prediction_length);
    
    clear train_samples
    
    gc = 1;                         % global iteration count
    [avg_g,avg_sqg] = deal([],[]);  % average gradients, average squared gradients (adam)
    tic;                            % start timer
    
    for epoch = 1:max_epochs
        
        locs = randperm(no_samples);% shuffle data
        
        for i = 1:batch_size:no_samples
            gc = gc+1;
            
            batch = [1:1+batch_size-1];
            batch(batch>no_samples) = [];
            
            xBatch = prepare_batch(xData(locs(batch)),'CTB',input_channels);
            yBatch = prepare_batch(yData(locs(batch)),'CTB',input_channels);
            
            [gradients,loss,net] = dlfeval(@model_gradients,net,xBatch,yBatch,dropout_rate);
            [net,avg_g,avg_sqg] = adamupdate(net,gradients,avg_g,avg_sqg,gc,learn_rate);
        end
        
        fprintf("epoch:\t%d, time_elapsed:\t%.2fs\tloss:\t%.2f\n",epoch,toc,loss);     
    end
    
%     fprintf("\nmodel evaluation\n\n");
%     evaluate_model(...
%         eval_samples{1},net,window_size,prediction_length,false,true,0,confidence_interval,heuristic_limit);
%     model{s} = net;    
%     fprintf("\n");
end

%% helper functions

function batch = prepare_batch(data,labels,no_channels)
    
    batch = cell2mat(data);
    batch = reshape(batch,[no_channels size(batch,2) size(batch,1)]);
    batch = gpudl(batch,labels);
end

function [gradients,loss,network] = model_gradients(network,xBatch,yBatch,dropout_rate)

    y = network.fw(xBatch(:,:,2:end),yBatch(:,:,1:end-1),dropout_rate);
    loss = sqrt(sum(power(y-yBatch(:,:,2:end),2),'all')/numel(y));
    
    gradients = dlgradient(loss,network.Learnables);
end