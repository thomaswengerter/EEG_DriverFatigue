clear

load(['AllChannelsData/FeaturesCh', num2str(1)])
Data = zeros(size(F.labels,1),30*(11+21)+1);
count = 1;
for i = 1:32
    if i == 2 || i == 27
        
    else
        
        load(['FeaturesCh', num2str(i)])
        DataSet = [F.PSDalpha1,F.PSDalpha2,F.PSDbeta1,F.PSDbeta2,F.alphaStd,F.betaStd,F.alphaMean,F.betaMean,...
        F.SampleEntropy,F.alphaSampleEntropy,F.betaSampleEntropy,F.FuzzyWPT];
        Data(:,(count-1)*(size(DataSet,2))+1 : count*(size(DataSet,2))) = DataSet;
        count = count+1;
    end
end
Data(:,end) = F.labels;
csvwrite(['FeaturesAllChann'],Data);
save('FeaturesAllChann','Data')