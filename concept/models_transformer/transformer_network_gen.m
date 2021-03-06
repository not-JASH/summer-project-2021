addpath("classes","functions","models");
clear_env();

%% define variables

datafile = "BTCUSDT_5M.txt";

% data generation varaiables

window_size = 12;
rate = 4;
no_samples = 1e4;
no_sets = 1;
prediction_length = 1;
time_before = 90*24*60/5;
time_after = 1*24*60/5;
no_bins = 1024;
cutoff = 0.01;

% trader variables

confidence_interval = 0.4;
heuristic_limit = 0;

% training variables

dropout_rate = 0.3;
batch_size = 512;
% learn_rate = 1e-5;
max_epochs = 100;
valid_split = 0.2;

% transformer variables

input_channels = 1; 
% target_size = 1; 
no_layers = 4;
d_model = 512;
no_heads = 8;
dff = 1048;

%% training loop

model = cell(no_sets,1);
bins = get_bins(datafile,no_bins,cutoff);
target_size = length(bins);
warmup_steps = 4e4;

data = binance_textload(datafile);

for s = 1:no_sets
    
    %   initialize network & generate training/evaluation data
    
    [net,weights] = transformer(...
        input_channels,target_size,no_layers,d_model,no_heads,dff,dropout_rate);
    
    sample.zeromean = data(:,5)-data(:,2); % close - open
    sample.data = data(:,5); %close
    
    %[train_samples,eval_samples] = get_samples(datafile,1,rate,time_before,time_after);
    [xData,yData,yRef,y] = subsample(sample,no_samples,window_size+1,prediction_length,bins);
    %[xData,yData,yBin,yRef,y] = subsample(train_samples{1},no_samples,window_size+1,prediction_length,bins);
    %[xVal,yVal] = subsample(eval_samples{1},valid_split*no_samples,window_size+1,prediction_length);

    clear train_samples
    
    gc = 1;                         % global iteration count
    [avg_g,avg_sqg] = deal([],[]);  % average gradients, average squared gradients (adam)
    tic;                            % start timer
    
    %for epoch = 1:max_epochs
    epoch = 0;
    fprintf("loop #%d\n",s);
    while true
        
        locs = randperm(no_samples);% shuffle data
        
        for i = 1:batch_size:no_samples
            gc = gc+1;
            epoch = epoch +1;
            
            batch = [1:1+batch_size-1];
            batch(batch>no_samples) = [];
            
            xBatch = prepare_batch(xData(locs(batch)));
            xBatch = xBatch/target_size;
            
            yBatch = prepare_batch(yData(locs(batch)));
            
            yrBatch = prepare_batch(yRef(locs(batch)));
            yrBatch = yrBatch/target_size;
                        
            [gradients,loss,accuracy,net] = dlfeval(@model_gradients,net,weights,xBatch,yBatch,yrBatch,dropout_rate);
            [weights,avg_g,avg_sqg] = adamupdate(weights,gradients,avg_g,avg_sqg,gc,get_learnrate(gc,warmup_steps));
        end
        
        fprintf("epoch:\t%d, time_elapsed:\t%.2fs\tloss:\t%.2f\taccuracy: %%%.2f\n",epoch,toc,loss,accuracy);     
    end
    
%     fprintf("\nmodel evaluation\n\n");
%     evaluate_model(...
%         eval_samples{1},net,window_size,prediction_length,false,true,0,confidence_interval,heuristic_limit);
%     model{s} = net;    
%     fprintf("\n");
end

%% helper functions

function learn_rate = get_learnrate(step_num,warmup_steps)
    learn_rate = power(step_num,-0.5)*min([power(step_num,-0.5),step_num*power(warmup_steps,-0.5)]);
end

function batch = prepare_batch(data)
    batch = data{1};
    for i = 2:length(data)
        batch = cat(3,batch,data{i});
    end
    batch = gpudl(batch,'CTB');
end

function [gradients,loss,accuracy,network] = model_gradients(network,weights,xBatch,yBatch,yrBatch,dropout_rate)
    persistent i
    if isempty(i)
        i = 0;
    end
    i = i+1;

    y = network.fw(xBatch(:,:,2:end),yrBatch(:,:,1:end-1),dropout_rate,weights);
    y = softmax(y);
    
    [~,x] = max(y);
    accuracy = 100*sum(gatext(reshape(yrBatch(:,:,2:end),size(x,2),size(x,3))) == gatext(reshape(x,size(x,2),size(x,3))),'all')/numel(x);
    
    %loss = sqrt(sum(power(y-yBatch(:,:,2:end),2),'all')/numel(y));
    loss = crossentropy(y,yBatch(:,:,2:end));

    %fprintf("batch loss: %.2f\taccuracy: %%%.2f\n",loss,accuracy);
    
    gradients = dlgradient(loss,weights);
end

function data = gatext(data)
    data = gather(extractdata(data));
end