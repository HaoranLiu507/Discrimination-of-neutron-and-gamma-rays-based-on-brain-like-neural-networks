% his file is used to label the TOF dataset, where 0 represents neutrons and 1 represents gamma
clc; clear; close all
warning off
load('gamma.mat'); 
load('neutron.mat');
data_gamma = cell(1, numel(temp_gamma));  
data_neutron = cell(1, numel(temp_neutron)); 
data_gamma{8} = temp_gamma{8}.data; 
data_neutron{8} = temp_neutron{8}.data;  
d1 = data_gamma{8};
d2 = data_neutron{8};


data = [ones(2215, 1); zeros(5476, 1)];
data = double(data);


