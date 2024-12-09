% This file is used to normalize the signal and further filter out abnormal signals
clc;clear;close all
warning off
siganl_t = load('5Mev.mat');
ss = (siganl_t.combined_data)';
normalized_signals = normalize(ss, 'range', [0, 1]);
normalized_signals = normalized_signals';
threshold = 0.1;
last_part = normalized_signals(:, 280:end); 
abnormal_indices = any(last_part > threshold, 2); 

% filtered_signals = normalized_signals(~abnormal_indices, :); 
% filtered_signals = filtered_signals';

ss = ss';
filtered_signals = ss(~abnormal_indices, :); 
filtered_signals = filtered_signals';