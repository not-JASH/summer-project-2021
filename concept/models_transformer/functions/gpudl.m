function data = gpudl(data,labels)
    data = gpuArray(dlarray(data,labels));
end