% Experiment of CCNN Model
clc;clear;close all
DATA_nomalized = load("5.0_signal_filter_norm.mat");
DATA_nomalized = (DATA_nomalized.filtered_signals)';
R_CCNN = CCNN_main(DATA_nomalized);

% Calculate FOM and fit histogram
[yc,miu,sigma,FOM] = Histogram_Fitting_and_Compute_FOM(R_CCNN');

% Calculate the accuracy of discrimination
R = mapminmax(R_CCNN,0,1);
R = R*200;
result_label = zeros(size(R_CCNN));  
result_label(R <= 90) = 1;
label_tof = load("5_label.mat");
label_tof = label_tof.filtered_label;
count = sum(label_tof == result_label);
accuracy_pd = count/length(label_tof);