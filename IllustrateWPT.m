%% WPT illustration

load('AllChannelsData/TrainingDataCh22')
f = 1e3;
t = 1/f:1/f:1;

x = Train.data(13,:);

[wpt,~,F] = modwpt(x,'TimeAlign',true);
contour(t,F.*(1/dt),abs(wpt).^2)
grid on
xlabel('Time (secs)')
ylabel('Hz')
title('Time-Frequency Analysis -- Undecimated Wavelet Packet Transform')