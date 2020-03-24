%% Pre-Processing of all EEG Channels
% Import CNT files -> EEGraw
% Apply Lowpassfilter and Highpassfilter -> EEG
% Calculate Power Density Spectrum PSD
% Split into segments of 1s duration
% Normalize for ML algorithm
% Define Labels: 0 awake / 1 fatigue
% Combine all patients and measurements to one large training data matrix
% (mix randomly)
close all;

%% INPUTS
% SELECT input settings here
path = 'C:\Users\Thomas\Documents\Uni\Master\UTS\Medical Instrumentation\MATLAB\figshare_Data\';
file = {'Normal state.cnt'; 'Fatigue state.cnt'}; %Select States to analyze
patients = 5; %SELECT Patient(s)
Channels = 22; %TP7: 22
tepoch = 1; %SELECT length of epoch (in seconds)
normalize = 1; %SELECT [1: normalize FFT spectrum of Epochs for AI training / 0: no normalization]

% Switch on/off (1/0) optional output plots 
plotfilt = 1; %PLOT transfer function of filters
plot1 = 1; %PLOT raw data TimeDomain (TD)
plot2 = 1; %PLOT raw data Power Density Spectra FrequencyDomain (FD)
plot3 = 1; %PLOT filtered data (TD)
plot4 = 1; %PLOT filtered data Power Density Spectra (FD)
plot5 = 1; %PLOT normalized FFT of one epoch (FD)

%% Global Variables
fs = 1e3; %sampling frequency 1kHz
dt = 1/fs; %time step

fhpf = 45; %Set lower Passband Freq (HPF)
flpf = 0.5; %Set upper Passband Freq (LPF)

lenEpoch = tepoch/dt; 
alpha = [8,13]; %Alpha freq band
beta = [13,30]; % Beta freq band
delta = [1:4];
theta = [4:8];

if plot1==1, fig1 = figure('Name','Raw Data'); end
if plot2==1, fig2 = figure('Name','Power Density Spectra'); end
if plot3==1, fig3 = figure('Name', 'Filtered Data'); end
if plot4==1, fig4 = figure('Name', 'Power Density Spectra Filtered Data'); end
if plot5==1, fig5 = figure('Name','(Normalized) FFT of epoch 10'); end


%% Start Preprocessing
EEGraw = {length(patients),length(file)};
EEG = {length(patients),length(file)};
EEGpsd = {length(patients),length(file)};
EpochsFD = {length(patients),length(file)};
SetLen = zeros(length(patients),length(file)); %Lengths of individual EEG datasets
numEpochs = zeros(length(patients),length(file));

for channel = Channels
    for patient = 1:length(patients)
        for meas = 1:length(file)
            %% Load Raw Data
            data = loadcnt([path, num2str(patients(patient)), '\', file{meas}],'dataformat','int32');
            EEGraw{patient,meas} = data.data(channel,:);

            SetLen(patient,meas) = length(EEGraw{patient,meas});
            % Plot RAW data TIME domain
            tseg = 10:dt:11-dt;
            if plot1
                figure(fig1)
                subplot(length(patients), length(file), (patient-1)*length(file)+meas)
                title(['10s-11s Data Segment Patient ', num2str(patients(patient)),' ', file{meas}]);
                plot(tseg,EEGraw{patient,meas}(1:length(tseg)))
                legend(['Raw EEG Segment Patient ', num2str(patients(patient)),' ', file{meas}])
                xlabel('t in s')
                ylabel('Amplitude in mV')
            end
            if plot2
                % Calculate Power Spectrum of RAW Data
                FD = fft(EEGraw{patient,meas}*1e-3);
                FD = FD(1:SetLen(patient,meas)/2+1);
                PSD = 1/(fs*SetLen(patient,meas))*abs(FD).^2;
                freq = 0 : fs/SetLen(patient,meas) : fs/2;
                % Plot Power Spectrum
                figure(fig2)
                subplot(length(patients), length(file), (patient-1)*length(file)+meas)
                title(['Power Spectrum Patient ', num2str(patients(patient)),' ', file{meas}])
                plot(freq,10*log10(PSD/1e-3))
                xlabel('Frequency in Hz')
                ylabel('PSD (dBm/Hz)');
            end

            %% Filtering Raw Data

            % 0.5Hz to 50Hz
            lpf = designfilt('lowpassfir', 'PassbandFrequency', fhpf, 'StopbandFrequency', fhpf*1.1, 'PassbandRipple', 0.01, 'StopbandAttenuation', 60, 'SampleRate', 1000,'DesignMethod','kaiserwin');
            hpf = designfilt('highpassfir', 'StopbandFrequency', 0.1*flpf, 'PassbandFrequency', flpf, 'StopbandAttenuation', 80, 'PassbandRipple', 0.01, 'SampleRate', 1000, 'DesignMethod', 'kaiserwin');

            % Show Filter transfer function
            if patient == 1 && plotfilt
                fvtool(lpf)
                fvtool(hpf)
            end
            % Filter EEG channel
            EEG{patient,meas} = filtfilt(lpf,EEGraw{patient,meas}); % apply LPF
            EEG{patient,meas} = filtfilt(hpf, EEG{patient,meas}); % apply HPF



            if plot3 == 1
                % plot FILTERED data TIME domain
                figure(fig3)
                subplot(length(patients), length(file), (patient-1)*length(file)+meas)
                title('filtfilt filtered Set');
                t = 0:dt:(length(EEG{patient,meas})-1)*dt;
                plot(t,EEG{patient,meas})
                legend(['Filtered EEG Patient ', num2str(patients(patient)),' ', file{meas}])
                xlabel('t in s')
                ylabel('Amplitude in mV')
            end

            % Calculate Power Spectrum of FILTERED Data
            FD = fft(EEG{patient,meas});
            FD = FD(1:SetLen(patient,meas)/2+1);
            PSD = 1/(fs*SetLen(patient,meas))*abs(FD).^2;
            EEGpsd{patient,meas} = PSD;
            freq = 0 : fs/SetLen(patient,meas) : fs/2;
            if plot4 == 1
                % Plot Power Spectrum
                figure(fig4)
                subplot(length(patients), length(file), (patient-1)*length(file)+meas)
                plot(freq,10*log10(PSD/1e-3))
    %             legend(['Filtered EEG PSD Patient ', num2str(patients(patient)),' ', file{meas}])
                xlabel('Frequency in Hz')
                ylabel('PSD (dBm/Hz)');
                title(['Filtered PSD Patient ', num2str(patients(patient)),' ', file{meas}])
                axis([0,60, -40, 60]);
    %             axis([0,100, -40, 60]);
            end


            %% Split time domain data into epochs and perform FFT
            numEpochs(patient,meas) = floor(length(EEG{patient,meas})/lenEpoch);
            EpochsTD{patient,meas} = zeros(length(numEpochs(patient,meas)), lenEpoch);
            EpochsFD{patient,meas} = zeros(length(numEpochs(patient,meas)), lenEpoch/2+1);
            for e = 1:numEpochs(patient,meas)
                epochData = EEG{patient,meas}(lenEpoch*(e-1)+1:lenEpoch*e); %extract epoch e
                EpochsTD{patient,meas}(e,:) = epochData;
                epochFD = fft(epochData); % apply FFT
                epochFD = abs(epochFD(1:lenEpoch/2+1)); % Absolute value of fft freq spectrum (complex Phase is dropped)
                if normalize, epochFD = epochFD/max(epochFD); end %NORMALIZE data
                EpochsFD{patient,meas}(e,:) = epochFD;
                if plot5 && e==10
                    figure(fig5)
                    subplot(length(patients), length(file), (patient-1)*length(file)+meas)
                    title(['Normalized absolute FFT Patient ', num2str(patients(patient)),' ', file{meas}])
                    freq = 0:fs/lenEpoch:fs/2;
                    plot(freq, epochFD)
                    xlabel('Frequency in Hz')
                    ylabel('Norm abs FFT');
                    legend(['Patient ', num2str(patients(patient)),' ', file{meas}])
                    grid on
                    if normalize, axis([0,50, 0, 1]); end
                    if ~normalize, axis([0,50, 0, 2000]); end
                end
            end
        end
    end



    %% Extract Alpha and Beta Band
    % Training Data contains full frequency spectrum [0,500]Hz.
    % From full frequency spectrum up to 500Hz, crop window of alpha and beta
    % waves and save it in seperate Training Data vector.
    TrainingDataTD = zeros(sum(numEpochs(:)),lenEpoch); % full spectrum [0,500]Hz
    TrainingDataFD = zeros(sum(numEpochs(:)),lenEpoch/2+1); % full spectrum [0,500]Hz
    TrainingDataAlpha = zeros(sum(numEpochs(:)), (alpha(2)-alpha(1))/(fs/lenEpoch)+1); % [8,13]Hz
    TrainingDataBeta = zeros(sum(numEpochs(:)), (beta(2)-beta(1))/(fs/lenEpoch)+1); % [13,30]Hz
    TrainingDataDelta = zeros(sum(numEpochs(:)), (delta(2)-delta(1))/(fs/lenEpoch)+1); % [1,4]Hz
    TrainingDataTheta = zeros(sum(numEpochs(:)), (theta(2)-theta(1))/(fs/lenEpoch)+1); % [4,8]Hz
    Labels = zeros(sum(numEpochs(:)),1);
    ecount = 1;

    for patient = 1:length(patients)
        for meas = 1:length(file)
            % Combine all patients into one TrainingData set
            TrainingDataTD(ecount:ecount+numEpochs(patient,meas)-1,:) = EpochsTD{patient,meas};
            TrainingDataFD(ecount:ecount+numEpochs(patient,meas)-1,:) = EpochsFD{patient,meas};

            dataAlpha = EpochsFD{patient,meas}(:, alpha(1)/(fs/lenEpoch)+1:alpha(2)/(fs/lenEpoch)+1); %extract window of alpha waves
            if normalize
                for l = 1:numEpochs(patient,meas)
                    dataAlpha(l,:) = dataAlpha(l,:)/max(dataAlpha(l,:));
                end             
            end
            TrainingDataAlpha(ecount:ecount+numEpochs(patient,meas)-1,:) = dataAlpha;

            dataBeta = EpochsFD{patient,meas}(:, beta(1)/(fs/lenEpoch)+1:beta(2)/(fs/lenEpoch)+1); %extract window of beta waves
            if normalize
                for l = 1:numEpochs(patient,meas)
                    dataBeta(l,:) = dataBeta(l,:)/max(dataBeta(l,:));
                end
            end
            TrainingDataBeta(ecount:ecount+numEpochs(patient,meas)-1,:) = dataBeta;

            dataDelta = EpochsFD{patient,meas}(:, delta(1)/(fs/lenEpoch)+1:delta(2)/(fs/lenEpoch)+1); %extract window of beta waves
            if normalize
                for l = 1:numEpochs(patient,meas)
                    dataDelta(l,:) = dataDelta(l,:)/max(dataDelta(l,:));
                end
            end
            TrainingDataDelta(ecount:ecount+numEpochs(patient,meas)-1,:) = dataDelta;

            dataTheta = EpochsFD{patient,meas}(:, theta(1)/(fs/lenEpoch)+1:theta(2)/(fs/lenEpoch)+1); %extract window of beta waves
            if normalize
                for l = 1:numEpochs(patient,meas)
                    dataTheta(l,:) = dataTheta(l,:)/max(dataTheta(l,:));
                end
            end
            TrainingDataTheta(ecount:ecount+numEpochs(patient,meas)-1,:) = dataTheta;


            if meas == 1, label = 0; else, label = 1; end %initialize boolean Labels
            Labels(ecount:ecount+numEpochs(patient,meas)-1,1) = label*ones(numEpochs(patient,meas),1);
            ecount = ecount+numEpochs(patient,meas);

        end
    end


    %% Create Output Struct
    Train = struct;
    Train.data = TrainingDataTD;
    Train.dataFD = TrainingDataFD;
    Train.alpha = TrainingDataAlpha;
    Train.beta = TrainingDataBeta;
    Train.labels = Labels;
    save(['TrainingDataCh',num2str(channel),'.mat'],'Train')
    % csvwrite('FullData.csv', Train.data);
    % csvwrite('Alpha.csv',Train.alpha);
    % csvwrite('Beta.csv', Train.beta);
    % csvwrite('Labels.csv', Train.labels);
end