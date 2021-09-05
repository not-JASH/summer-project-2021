function [train_sample,eval_sample] = get_sample(data,rate,time_before,time_after)
    xo = randi([time_before, length(data)-time_after],1);
   
    train_sample = pois([data(xo-time_before:xo,5),data(xo-time_before:xo,2)],rate);
    eval_sample = pois([data(xo:xo+time_after,5),data(xo:xo+time_after,2)],rate);
end