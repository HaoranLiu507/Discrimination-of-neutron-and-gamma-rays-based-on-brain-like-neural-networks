% Experiment of CCNN Model
clc;clear;close all
DATA_nomalized = load("5.0_signal_filter_norm.mat");
DATA_nomalized = (DATA_nomalized.filtered_signals)';
R_RCNN = RCNN_main(DATA_nomalized);

% Calculate FOM and fit histogram
[yc,miu,sigma,FOM] = Histogram_Fitting_and_Compute_FOM(R_RCNN');

% Calculate the accuracy of discrimination
R = mapminmax(R_RCNN, 0, 1);
R = R * 200;
result_label = zeros(size(R_RCNN));  
result_label(R <= 80) = 1;
label_tof = load("5_label.mat");
label_tof = label_tof.filtered_label;
count = sum(label_tof == result_label);

accuracy_pd = count/length(label_tof);