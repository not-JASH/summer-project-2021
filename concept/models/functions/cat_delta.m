function data = cat_delta(data,bins)

    for pt = 1:length(data)
        
        if data(pt) < bins(1)
            data(pt) = 1;
            continue
        end
        
        for bin = 1:length(bins)-1
            if data(pt) >= bins(bin) && data(pt) < bins(bin+1)
                data(pt) = bin;
                break
            end
        end
        
        if data(pt) >= bins(end)
            data(pt) = length(bins);
        end
    end
end