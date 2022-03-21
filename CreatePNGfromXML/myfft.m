function myfft(signal,fs)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
figure
% FFT = fft(signal);
% L = length(signal);
% P2 = abs(FFT/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = fs*(0:L/2)/L;
% plot(f,P1)


NFFT = 2 ^ nextpow2(length(signal));  %compute FFT length depends on the signal length
Y = fft(signal,NFFT);  %compute the fft of the noisy signal
Y = Y(1:NFFT/2);  %we only need a one sided fft plot
P1 = 1/NFFT*abs(Y); %calculate the magnitude and normalize the spectrum
P1 = P1 - mean(P1);
f = (0:NFFT/2-1)*fs/NFFT;
plot(f,P1)


end