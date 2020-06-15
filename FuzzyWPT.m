%% Fuzzy Entropy

load('AllChannelsData/TrainingDataCh1')
data = zeros(size(Train.data,1)*30, size(Train.data,2));

epochs = size(Train.data,1);
count = 1;
for i = 1:32
    if i ~= 2 && i ~= 27
        load(['AllChannelsData/TrainingDataCh', num2str(count)])
        if size(Train.data,1)~= epochs
            fprintf('size not matching')
            break
        end
        data((count-1)*epochs+1:count*epochs, :) = Train.data;
        count = count+1;
    end
end
fprintf('Data extracted. Start Fuzzy WPT...')


%% Khushaba Fuzzy WPT feature extraction
wtlvl = 5;
trainSets = size(data,1);

% 1: Apply WPT to the labeled dataset
WTF = {};
wtf = zeros(size(data,1),sum(2.^(0:wtlvl-1)));
idx = zeros(sum(2.^(0:wtlvl-1)),2);
for r = 1:trainSets %size(data,1)
    wpt = wpdec(data(r,:), wtlvl, 'db4'); %db4 (Wang.2019) or sym8 (Substractive Fuzzy) 
    
    for lvl = 0:wtlvl-1
        for node = 0:2^lvl-1
            % WT Feature: Energy of each DWT node
            WTF{lvl+1,node+1} = log10( 1/(size(data,2)/2^lvl)*sum(wpcoef(wpt, [lvl,node]).^2) );
            %collect in feature vector wtf
            wtf(r,sum(2.^(0:lvl-1))+ node+1) = WTF{lvl+1,node+1};
            idx(sum(2.^(0:lvl-1))+ node+1,:) = [lvl,node];
        end
    end
end

% 2: Fuzzy Entropy and MI calculation of Feature Vectors (rows of wtf)
[I_Cf, I_Cff, I_ff, H_f, H_ff, H_C] = Fuzzy_MI([wtf(1:trainSets,:),repmat(Train.labels,30,1)]);  
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
WTpower = zeros(size(data,1),length(Features));
for feat = 1:length(Features)
    % find Wavelet Transform Features with highest Fuzzy Entropy
    % -> wtf(Fidx), find(Fidx,idx)
    logic = (Fidx(feat,1) == idx(:,1) & Fidx(feat,2) == idx(:,2));
    WTpower(:,feat) = wtf(:,logic);
end

