function getWorkers(nWorkers)
    if isempty(gcp('nocreate'))||gcp('nocreate').NumWorkers ~= nWorkers
        delete(gcp('nocreate'));
        parpool(nWorkers);
    end
end
