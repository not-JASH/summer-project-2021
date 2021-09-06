function evaluate_model(sample,model,window_size,prediction_length,verbose)
    trader = interface(window_size,prediction_length,0);
    trader.model = model;
    
    [delta,entry,wins,losses,total] = deal(0,0,0,0,1);
    [wX,wClose] = deal(zeros(1,window_size),zeros(1,window_size));
    
    for i = 1:length(sample.data)
        %if rem(i,1440) == 0
        %    fprintf("\n day %d",rem(i,1440)+1);
        %end

        [wX,wClose] = deal(cycle(wX,sample.zeromean(i)),cycle(wClose,sample.data(i)));
        if wX(1) == 0
            continue
        end
        
        [ls,exit] = trader.iter(wX,wClose);
        if ~isempty(ls)
            if entry == 0
                entry = exit;
            elseif ls
                temp = entry-exit;
                temp = temp/entry;
                
                if temp > 0
                    wins = wins+1;
                else 
                    losses = losses + 1;
                end 
                
                total = total*(1+temp);
                delta = delta + temp;
                entry = exit;
                if verbose
                    fprintf("minute %d, long, delta: %.2f, accuracy: %.2f\n",i,delta,100*wins/(wins+losses));
                end
            elseif ~ls
                temp = exit-entry;
                temp = temp/entry;
                
                if temp > 0
                    wins = wins+1;
                else 
                    losses = losses +1;
                end
                
                total = total*(1+temp);
                delta = delta+temp;
                entry = exit;
                
                if verbose
                    fprintf("minute %d, short, delta: %.2f, accuracy: %.2f\n",i,delta,100*wins/(wins+losses));
                end
            end
        end        
    end
    
    fprintf("Delta: %.2f\nAccuracy: %.2f\nTotal Trades: %d\nexp total: %.2f\n",delta,100*wins/(wins+losses),wins+losses,total);
end
    
    
