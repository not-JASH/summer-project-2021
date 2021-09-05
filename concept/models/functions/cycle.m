function data = cycle(data,npt)
    data = data(2:end);
    data = [data;npt];
end