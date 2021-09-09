%{
    model weights will be determined 
    aim is to maximize delta and exp_total resulting from evaluation
    function
    
    instead of accurately assesing whether or not a trade is adequate.

%}
addpath("classes","functions","models");

model = [
    sequenceInputLayer(1,"Name","sequence")
    fullyConnectedLayer(512,"Name","fc_1")
    reluLayer("Name","relu_1")
    bilstmLayer(128,"Name","bilstm_1")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(256,"Name","fc_2")
    bilstmLayer(128,"Name","bilstm_2","OutputMode","last")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(2,"Name","fc_3")
    softmaxLayer("Name","softmax")
    ];

model_graph = layerGraph(model);

rate = 30;
no_samples = 1e3;
eval_duration = 1*24*60;

[train_samples,~] = get_samples("BTCUSDT.txt",no_samples,rate,eval_duration,0);

learn_rate = 1e-3;
no_epochs = 30;
net = dlnetwork(model_graph);
avg_g = [];avg_sqg = [];
loss = dlarray([]);

gc = 1;
for i = 1:no_epochs
    for j = 1:no_samples
        gc = gc+1;    
        sloc = randperm(no_samples);
        [gradients,loss,net] = dlfeval(@model_gradients,net,train_samples{sloc(j)});
        [net,avg_g,avg_sqg] = adamupdate(net,gradients,avg_g,avg_sqg,gc,learn_rate);
        
        fprintf("\nloss: %.2f\n\n",loss);
    end
end



function [gradients,loss,network] = model_gradients(network,sample)


    window_size = 60;
    prediction_length = 0;
    
    [delta,wins,losses,total,~,network] = ...
        evaluate_model(sample,network,window_size,prediction_length,false,false,0);
    
    % :)
    traced_array = predict(network,dlarray(rand(1,window_size),'CT'));    
    loss = calculate_loss(delta,wins,losses,total)*sum(traced_array);
        
    gradients = dlgradient(loss,network.Learnables);

    %network.state = ?
end

function loss = calculate_loss(delta,wins,losses,total)
    loss = 1/total;
    if wins == 0 && losses == 0 && delta == 0
        loss = loss + rand(1);
    end
end






