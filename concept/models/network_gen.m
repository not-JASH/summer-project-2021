addpath("classes","functions","models");

window_size = 60;
no_samples = 1e4;
no_sets = 10;
prediction_length = 1;
time_before = 24*60;
time_after = 4*60;

[train_samples,eval_samples] = get_samples("BTCUSDT.txt",no_sets,30,time_before,time_after);

for i = 1:no_sets
    [xData,yData] = subsample(train_samples{i},no_samples,window_size,prediction_length);
    [xVal,yVal] = subsample(eval_samples{i},0.2*no_samples,window_size,prediction_length);
end