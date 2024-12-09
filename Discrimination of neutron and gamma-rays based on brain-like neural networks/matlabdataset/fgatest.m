% Experiment of FGA
clc;clear;close all
DATA_nomalized = load("2.0_signal_filter_norm.mat");
DATA_nomalized = (DATA_nomalized.filtered_signals)';
R_FGA = FGA(DATA_nomalized);

% Calculate FOM and fit histogram
[yc,miu,sigma,FOM] = Histogram_Fitting_and_Compute_FOM(R_FGA');

% Calculate the accuracy of discrimination
R = mapminmax(R_FGA, 0, 1);
R = R * 200;
result_label = zeros(size(R_FGA));  
result_label(R <= 45) = 1;
label_tof = load("2_label.mat");
label_tof = label_tof.filtered_label;
count = sum(label_tof == result_label);
accuracy_pd = count/length(label_tof);