clear
clc
close all

% File import
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[file, folder] = uigetfile('*.XML'); %Choose an XML file
filename = file(1:end-4);


% Handles some directory structure pecularities in some systems
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fslash = strfind(folder,'/');
if isempty(fslash)==1
    dirdemarc = '\';
else
    dirdemarc = '/';
end

slashcheck = strcmp(folder(end),dirdemarc);
if slashcheck == 0
    folder = [folder dirdemarc];
end
if isempty(fslash)==1
    saveXMLpath = '..\Data\DataXML\';
else
    saveXMLpath = '../Data/DataXML/';
end

filexml = strcat(folder,file);
copyfile(filexml,pwd)

% Read xml and decode leads pre-processing info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
leads_pp = decoding(file,folder);

readxml = xml2struct(file);

% per ottenere risoluzione in ampiezza
amp_uV = str2double(readxml.restingecgdata.dataacquisition.signalcharacteristics.resolution.Text);
overall_gain = str2double(readxml.restingecgdata.reportinfo.reportgain.amplitudegain.overallgain.Text);
amp_factor = amp_uV/1000*overall_gain; % in mm

% per ottenere risoluzione temporale
sampleRate = str2double(readxml.restingecgdata.dataacquisition.signalcharacteristics.samplingrate.Text);
time_gain = str2double(readxml.restingecgdata.reportinfo.reportgain.timegain.Text);
time_factor = 1/sampleRate*time_gain; % in mm

leads_pp = leads_pp(1:5000,:);
leads_mm = leads_pp * amp_factor;
leads_mV = leads_mm / overall_gain;
% for i = 1:12
%     eval(sprintf('lead%d_pp = leads_mm(:,%d)',i,i));
% end


% 12-Lead Plot pre-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sigplot_pp = figure('numbertitle','off','name','12-Lead Plots pre-processing');
pos = [1 3 5 7 9 11 2 4 6 8 10 12];
set(gcf,'Position', get(0,'ScreenSize'));
labels = ["I", "II","III","aVR","aVL","aVF", "V1", "V2", "V3", "V4", "V5", "V6"];

plot12leads(pos,leads_mm,time_factor,labels)

% figname = [filename '_preproc.png'];
% exportgraphics(sigplot_pp, fullfile(finalpath,figname),'Resolution',600);

% Processing

% denoising: notch at 50Hz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fn = sampleRate/2;  % Nyquist frequency
numHarmonics = 3;   % 50, 100, 150, 200
lineFreq = 50;      % Hz
for i = 1:12
%     myfft(leads_mm(:,i),sampleRate)
    % % notch
    w01 = lineFreq/Fn;
    bw1 = w01/35;
    [num1,den1] = iirnotch(w01,bw1);
    leads_filt(:,i) = filtfilt(num1,den1,leads_mm(:,i));
    w02 = lineFreq*2/Fn;
    bw2 = w02/35;
    [num2,den2] = iirnotch(w02,bw2);
    leads_filt(:,i) = filtfilt(num2,den2,leads_filt(:,i));
    w03 = lineFreq*3/Fn;
    bw3 = w03/35;
    [num3,den3] = iirnotch(w03,bw3);
    leads_filt(:,i) = filtfilt(num3,den3,leads_filt(:,i));
%     for fq = ((0:numHarmonics)+1) * lineFreq
%         Fl = fq + [-1, 1]; % notch around Fl. Could try [-2, 2] if too tight
%         [z,p,k] = butter(6, (Fl/Fn), 'stop');
%         sos = zp2sos(z,p,k);
%         leads_filt(:,i) = filtfilt(sos,1,leads_mm(:,i)); % assumes data is [time x ... dimensions]
%         % overwrites data, and filters sequentially for each notch
%     end
%     myfft(leads_filt(:,i),sampleRate)
end

%%%
% for i = 1:12
%     eval(sprintf('lead%d_filt = filtfilt(num,den,lead%d_pp)',i,i));
% end
% leads_filt1 = [lead1_filt lead2_filt lead3_filt lead4_filt lead5_filt lead6_filt lead7_filt lead8_filt lead9_filt lead10_filt lead11_filt lead12_filt];


% detrending
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

leads = detrending(leads_filt);

% 12-Lead Plot post-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sigplot = figure('numbertitle','off','name','12-Lead Plots');
pos = [1 3 5 7 9 11 2 4 6 8 10 12];
set(gcf,'Position', get(0,'ScreenSize'));
labels = ["I", "II","III","aVR","aVL","aVF", "V1", "V2", "V3", "V4", "V5", "V6"];

plot12leads(pos,leads,time_factor,labels)

if isempty(fslash)==1
    savePNGpath = '..\Data\DataPNG\';
else
    savePNGpath = '../Data/DataPNG/';
end
figname = [filename '.png'];
exportgraphics(sigplot, fullfile(savePNGpath,figname),'Resolution',600);


% %%
% plot1lead(leads_mm,time_factor,labels)

%
% Save the 12-Lead data to a CSV file for general analytical purposes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [savefile,savefolder] = uiputfile('*.csv');

if isempty(fslash)==1
    saveCSVpath = '..\Data\DataCSV\';
else
    saveCSVpath = '../Data/DataCSV/';
end
savefile = [saveCSVpath filename '.csv'];
writematrix(leads,savefile);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
