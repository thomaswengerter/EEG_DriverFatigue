%% Feature Extraction
% load Time Domaingdata in struct 'Train'
% 0 Filter alpha beta time signal
% 1 Spectra extraction alpha beta
% 2 Std deviation and mean of alpha beta time signal
% 3 Sample Entropy
% 4 Spectral Entropy
% 5 Khushaba Fuzzy WPT feature extraction
% save Features in 'F' in file 'FeaturesCh1'
close all
clear

fs = 1000; %sampling freq in Hz
alpha = [8 13]; %alpha freq band
beta = [13 30]; %beta freq band

for i = 1:12 %patient numbers
    load(['AllChannelsData/TrainingDataCh',num2str(i)]) %load existing Train dataset
    if Train.data(10,10)~=0 %check if dataset is valid
        lenEpoch = size(Train.data,2);
        
        %% 0 Filter alpha beta time signal
        alphaTD=zeros(size(Train.data));
        betaTD=zeros(size(Train.data));
        for set = 1:size(Train.data,1)
            [wt,freq] = cwt(Train.data(set,:),fs);
            alphaTD(set,:) = icwt(wt,freq,alpha,'SignalMean',mean(Train.data(set)));
            betaTD(set,:) = icwt(wt,freq,beta,'SignalMean',mean(Train.data(set)));
        end


        %% 1 Spectra extraction alpha beta
        % PSD of FFT segments alpha1, alpha2, beta1, beta2
        Alpha = Train.dataFD(:, alpha(1):alpha(2)); %extract window of alpha waves
        Beta = Train.dataFD(:, beta(1):beta(2)); %extract window of beta waves


        PSDalpha = 10*log(sum(1/(fs*size(Alpha,2)).*abs(Alpha).^2,2));
        PSDbeta = 10*log(sum(1/(fs*size(Beta,2)).*abs(Beta).^2,2));
        %alpha1 [8 10]      alpha2 [11 13]
        PSDalpha1 = 10*log(sum(1/(fs*size(Alpha(:,1:floor(size(Alpha,2)/2)),2)).*abs(Alpha(:,1:floor(size(Alpha,2)/2))).^2,2));
        PSDalpha2 = 10*log(sum(1/(fs*size(Alpha(:,floor(size(Alpha,2)/2)+1:end),2)).*abs(Alpha(:,floor(size(Alpha,2)/2)+1:end)).^2,2));
        % beta1 [13 21]     beta2 [22 30]
        PSDbeta1 = 10*log(sum(1/(fs*size(Beta(:,1:floor(size(Beta,2)/2)),2)).*abs(Beta(:,1:floor(size(Beta,2)/2))).^2,2));
        PSDbeta2 = 10*log(sum(1/(fs*size(Beta(:,floor(size(Beta,2)/2)+1:end),2)).*abs(Beta(:,floor(size(Beta,2)/2)+1:end)).^2,2));



        %% 2 Std deviation and mean of alpha beta time signal
        alphaStd = zeros(size(Train.data,1),1);
        betaStd = zeros(size(Train.data,1),1);
        alphaMean = zeros(size(Train.data,1),1);
        betaMean = zeros(size(Train.data,1),1);
        for set = 1:size(Train.dataFD,1)
            alphaStd(set)=std(alphaTD(set,:));
            betaStd(set)=std(betaTD(set,:));
            alphaMean(set)=mean(alphaTD(set,:));
            betaMean(set)=mean(betaTD(set,:));
        end


        %% 3 Sample Entropy
        % SaEn on the unfiltered TD signal
        SaEn = zeros(size(Train.data,1),1);
        for sample = 1:size(Train.data,1)
            SaEn(sample,:) = SampEn(2,0.2*std(Train.data(sample,:)),Train.data(sample,:));
        end

        % SaEn on alpha beta band
        alphaSaEn = zeros(size(alphaTD,1),1);
        betaSaEn = zeros(size(betaTD,1),1);
        for sample = 1:size(Train.data,1)
            alphaSaEn(sample,:) = SampEn(2,0.2*std(alphaTD(sample,:)),alphaTD(sample,:));
            betaSaEn(sample,:) = SampEn(2,0.2*std(betaTD(sample,:)),betaTD(sample,:));
        end

        %% 4 Spectral Entropy
        % individually alpha beta??
        SpecEn = zeros(size(Train.dataFD,1),30);
        t = 1/fs:1/fs:1;
        for sample = 1:size(Train.dataFD,1)
            [specEn,te] = pentropy(Train.data(sample,:),t);
            SpecEn(sample,:) = specEn';
        end  

        %% 5 Khushaba Fuzzy WPT feature extraction
        wtlvl = 5;
        trainSets = size(Train.data,1);

        % 1: Apply WPT to the labeled dataset
        WTF = {};
        wtf = zeros(size(Train.data,1),sum(2.^(0:wtlvl-1)));
        idx = zeros(sum(2.^(0:wtlvl-1)),2);
        for r = 1:trainSets %size(Train.data,1)
            wpt = wpdec(Train.data(r,:), wtlvl, 'db4'); %db4 wavelet (Wang 2019)

            for lvl = 0:wtlvl-1
                for node = 0:2^lvl-1
                    % E_WPT Feature vector: Energy of each DWT node
                    WTF{lvl+1,node+1} = log10( 1/(size(Train.data,2)/2^lvl)*sum(wpcoef(wpt, [lvl,node]).^2) );
                    %collect in feature vector wtf and save leaf indeces in idx
                    wtf(r,sum(2.^(0:lvl-1))+ node+1) = WTF{lvl+1,node+1};
                    idx(sum(2.^(0:lvl-1))+ node+1,:) = [lvl,node];
                end
            end
        end

        % 2: Fuzzy Entropy and MI calculation of Feature Vectors (rows of wtf)
        [I_Cf, I_Cff, I_ff, H_f, H_ff, H_C] = Fuzzy_MI([wtf(1:trainSets,:),Train.labels(1:trainSets)]);  
        fMI = I_Cf./H_f; %normalized Fuzzy MI

        % 3: find WPT nodes with max Fuzzy MI
        [fMI,ix] = sort(fMI,'descend');
        idxFuzzy = idx(ix,:);

        % 4-7: Reduce dimensionality of MI features
        % find Features with highest MI I_Cf (most relevant for classification!)
        Features = zeros(1,length(fMI));
        Fidx = zeros(length(fMI),2);
        Qu = fMI;
        Quidx = idxFuzzy;

        items = 0;
        FWPT = {};
        while ~isempty(Qu)
            % extract node/feature/column with highest MI
            items = items+1;
            node = Quidx(1,:);
            Features(items) = Qu(1);
            Fidx(items,:) = Quidx(1,:);
            Qu(1) = [];
            Quidx(1,:) = [];
            %check the queue for father or daughter nodes and remove from queue
            rowfather = (Quidx(:,1) == node(1)-1 & Quidx(:,2) == floor(node(2)/2));
            if sum(rowfather)>=1
                %reduce feature dimensionality
                Qu(rowfather)=[];
                Quidx(rowfather,:) = [];
                Features(end) = [];
                Fidx(end,:) = [];
            end
            rowdaughter1 = (Quidx(:,1) == node(1)+1 & Quidx(:,2) == 2*node(2));
            rowdaughter2 = (Quidx(:,1) == node(1)+1 & Quidx(:,2) == 2*node(2)+1);
            if sum(rowdaughter1)>=1
                %reduce feature dimensionality
                Qu(rowdaughter1)=[];
                Quidx(rowdaughter1,:) = [];
                rowdaughter2(rowdaughter1) = [];
                Features(end) = [];
                Fidx(end,:) = [];
            end
            if sum(rowdaughter2)>=1
                %reduce feature dimensionality
                Qu(rowdaughter2)=[];
                Quidx(rowdaughter2,:) = [];
                Features(end) = [];
                Fidx(end,:) = [];
            end


        end

        % Extract FWPT subspaces
        WTpower = zeros(size(Train.data,1),length(Features));
        for feat = 1:length(Features)
            % find Wavelet Transform Features with highest Fuzzy Entropy
            % -> wtf(Fidx), find(Fidx,idx)
            logic = (Fidx(feat,1) == idx(:,1) & Fidx(feat,2) == idx(:,2));
            WTpower(:,feat) = wtf(:,logic);
        end


        %% Save Features in Struct F
        F = struct;
        F.PSDalpha = PSDalpha;
        F.PSDbeta = PSDbeta;
        F.PSDalpha1 = PSDalpha1;
        F.PSDalpha2 = PSDalpha2;
        F.PSDbeta1 = PSDbeta1;
        F.PSDbeta2 = PSDbeta2;
        F.alphaStd = alphaStd;
        F.betaStd = betaStd;
        F.alphaMean = alphaMean;
        F.betaMean = betaMean;
        F.SampleEntropy = SaEn;
        F.alphaSampleEntropy = alphaSaEn;
        F.betaSampleEntropy = betaSaEn;
        F.SpecEntropy = SpecEn;
        F.labels = Train.labels;
        F.FuzzyWPT = WTpower;
        F.DataSet = [PSDalpha1,PSDalpha2,PSDbeta1,PSDbeta2,alphaStd,betaStd,alphaMean,betaMean,...
            SaEn,alphaSaEn,betaSaEn,Train.labels];

        save(['AllChannelsData/FeaturesCh',num2str(i),'.mat'],'F')


        % csvwrite(['SpectralEntropyCh',num2str(i)],[SpecEn,Train.labels]);
        % csvwrite(['FuzzyWPTCh',num2str(i)],[WTpower,Train.labels]);
        % csvwrite(['DataSetCh',num2str(i)],DataSet);

    end
end



% cwt(Train.data(1,:),fs)
% [wt,f,coi] = cwt(Train.data(1,:),fs);
% 
% % set values in cone of influence to 0
% wt0 = wt;
% for idx = 1:length(coi)
%     coiVec = wt0(:,idx)<=coi(idx);
%     wt0(coiVec,idx) = 0;
% end

%% 
% figure
% imagesc(0:lenEpoch,f,abs(wt(end:-1:1,:)))
% set(gca,'YDir','normal')
% set(gca,'YTickLabels', flip(f));
% % set(gca,'YScale','log')
% xlabel('Time in s')
% ylabel('Freqency in Hz')
% colorbar
% 
% figure
% handle = pcolor(0:1/fs:lenEpoch-1/fs,f,abs(wt(end:-1:1,:)));
% handle.FaceColor = 'interp';
% xlabel('Time in s')
% ylabel('Freqency in Hz')
% colorbar
