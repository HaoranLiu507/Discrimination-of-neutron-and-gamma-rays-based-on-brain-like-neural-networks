% Experiment of PCNN Model
clc;clear;close all
DATA_nomalized = load("5.0_signal_filter_norm.mat");
DATA_nomalized = (DATA_nomalized.filtered_signals)';
R_PCNN = PCNN_main(DATA_nomalized);

% Calculate FOM and fit histogram
[yc,miu,sigma,FOM] = Histogram_Fitting_and_Compute_FOM(R_PCNN');

% Calculate the accuracy of discrimination
R = mapminmax(R_PCNN, 0, 1);
R = R * 200;
result_label = zeros(size(R_PCNN));  
result_label(R <= 65) = 1;
label_tof = load("2_label.mat");
label_tof = label_tof.filtered_label;
count = sum(label_tof == result_label);

accuracy_pd = count/length(label_tof);