% Experiment of SDCC Model
clc;clear;close all
DATA_nomalized = load("2.0_signal_filter_norm.mat");
DATA_nomalized = (DATA_nomalized.filtered_signals)';
R_SDCC = SDCC(DATA_nomalized);
R_min = min(R_SDCC); 
R_max = max(R_SDCC);  
if R_max > R_min
    R_SDCC_normalized = (R_SDCC - R_min) / (R_max - R_min);
else
    R_SDCC_normalized = zeros(size(R_SDCC));  
end
R_SDCC_normalized = R_SDCC_normalized*200;

% Calculate FOM and fit histogram
[yc,miu,sigma,FOM] = Histogram_Fitting_and_Compute_FOM(R_SDCC);

% Calculate the accuracy of discrimination
result_label = zeros(size(R_SDCC_normalized));  
result_label(R_SDCC_normalized <= 80) = 1;
label_tof = load("2_label.mat");
label_tof = (label_tof.filtered_label)';
count = sum(label_tof == result_label);
accuracy_pd = count/length(label_tof);