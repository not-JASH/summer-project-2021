%% Load data
clc;clear all
data = binance_textLoad("./BTCUSDT.txt");
data = data(:,[5,2]);

%% Process Variables.
rate = 30;
nSamples = 100;

%% Initialize samples & prepare environment.
samples = cell(100,1);
limits = round(linspace(1,size(data,1),nSamples+1)');

for i = 1:nSamples
    samples{i} = pois(data(limits(i):limits(i+1),:),rate);
end

getWorkers(6);

%% Optimize on each sample.

parfor i = 1:nSamples
    samples{i} = samples{i}.addPoints;
end

%% Cleanup and save.
for i = 1:nSamples
    samples{i}.points(1) = [];
    samples{i}.points(end) = [];
end

save('samples.mat');

%% Helper Functions 

