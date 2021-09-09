function [delta, wins, losses, total,trade_list,model] = evaluate_model(sample,model,window_size,prediction_length,verbose,plots,offset,confidence_interval,heuristic_limit)
    
    trader = interface(window_size,prediction_length,offset,confidence_interval,heuristic_limit);
    trader.model = model;
    
    [delta,entry,wins,losses,total] = deal(0,0,0,0,1);
    [wX,wClose] = deal(zeros(1,window_size),zeros(1,window_size));
    %[long/short entry exit delta]
    trade_list = {};
    trade_duration = 0;
    
    trade_fee = 0.001; %0.1-0.5%
    
    for i = 1:length(sample.data)
        %if rem(i,1440) == 0
        %    fprintf("\n day %d",rem(i,1440)+1);
        %end

        [wX,wClose] = deal(cycle(wX,sample.zeromean(i)),cycle(wClose,sample.data(i)));
        if wX(1) == 0
            continue
        end
        
        [trader,ls,exit,prediction] = trader.iter(wX,wClose);
        if plots
            yyaxis left
            plot(prediction)
            yline(0.5)
            yyaxis right
            plot(wClose)
            %w = waitforbuttonpress;
            f = getframe;          
        end       
        
        trade_duration = 1+ trade_duration;
        if ~isempty(ls)
            if entry == 0
                trade_duration =0;
                entry = exit;
            elseif ls
                temp = entry-exit;
                temp = temp/entry;
                temp = temp - trade_fee;
                
                if temp > 0
                    wins = wins+1;
                else 
                    losses = losses + 1;
                end 
                
                total = total*(1+temp);
                delta = delta + temp;
                trade_list = add_element(trade_list,[0,trade_duration,entry,exit,temp]);
                entry = exit;
                trade_duration = 0;
                if verbose
                    fprintf("minute %d, long, delta: %.2f, accuracy: %.2f\n",i,delta,100*wins/(wins+losses));
                end
            elseif ~ls
                temp = exit-entry;
                temp = temp/entry;
                temp = temp - trade_fee;
                
                if temp > 0
                    wins = wins+1;
                else 
                    losses = losses +1;
                end
                
                total = total*(1+temp);
                delta = delta+temp;
                trade_list = add_element(trade_list,[1,trade_duration,entry,exit,temp]);
                entry = exit;                
                trade_duration = 0;
                if verbose
                    fprintf("minute %d, short, delta: %.2f, accuracy: %.2f\n",i,delta,100*wins/(wins+losses));
                end
            end
        end        
    end
    
    fprintf("Delta: %.2f\nAccuracy: %.2f%%\nTotal Trades: %d\nexp total: %.2f\n",delta,100*wins/(wins+losses),wins+losses,total);
    
    model = trader.model;    
       
    function data = add_element(data,element)
       data = [data,element];
    end   
end
    
    
