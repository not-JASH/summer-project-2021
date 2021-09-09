function [training_samples,eval_samples] = get_samples(filename,no_samples,rate,time_before,time_after)
    data = binance_textload(filename);
    
    training_samples = cell(no_samples,1);
    eval_samples = cell(no_samples,1);
    
    for i = 1:no_samples
        [training_samples{i},eval_samples{i}] = get_sample(data,rate,time_before,time_after);
    end
    
    if no_samples == 1
        training_samples{1} = training_samples{1}.addPoints;
        if time_after ~= 0
            eval_samples{1} = eval_samples{1}.addPoints;
        end
    else
        get_workers(min([6,no_samples]));
        parfor i = 1:no_samples
            training_samples{i} = training_samples{i}.addPoints;
            if time_after ~= 0
                eval_samples{i} = eval_samples{i}.addPoints;
            end
        end
    end
end