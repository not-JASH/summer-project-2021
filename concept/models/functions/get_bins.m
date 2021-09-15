function bins = get_bins(datafile,no_bins,cutoff)
    data = binance_textload(datafile);
    
    % timestamp open high low close
    % data = close - open
    data = data(:,5) - data(:,2);
    
    [n,bins] = hist(data,no_bins);
    
    n = n/sum(n);
    bins(n < cutoff) = [];
end