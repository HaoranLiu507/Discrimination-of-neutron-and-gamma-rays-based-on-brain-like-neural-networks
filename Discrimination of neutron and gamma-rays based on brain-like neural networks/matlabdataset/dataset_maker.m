% This file is used to convert signals stored in txt format into array format
clc;clear;close all
warning off
data_root = 'G:\matlab_ps';
data.gamma = struct();
data.neutron = struct();
temp_gamma = cell(1, numel(1.5:0.5:5.5));
temp_neutron = cell(1, numel(1.5:0.5:5.5));
parfor i = 1:numel(1.5:0.5:5.5)
    mev = 1.5 + (i-1) * 0.5;  
    gamma_file = fullfile(data_root, sprintf('NE213A_%.1fMev', mev), sprintf('gamma_%.1fMev.txt', mev));
    neutron_file = fullfile(data_root, sprintf('NE213A_%.1fMev', mev), sprintf('neutron_%.1fMev.txt', mev));

    field_name = strrep(sprintf('data_%.1fMev', mev), '.', '_');


    gamma_data = [];
    neutron_data = [];

    temp_gamma{i} = struct('field_name', field_name, 'data', gamma_data);
    temp_neutron{i} = struct('field_name', field_name, 'data', neutron_data);
end


for i = 1:numel(temp_gamma)
    if ~isempty(temp_gamma{i}.data)  
        data.gamma.(temp_gamma{i}.field_name) = temp_gamma{i}.data;
    end
    if ~isempty(temp_neutron{i}.data)  
        data.neutron.(temp_neutron{i}.field_name) = temp_neutron{i}.data; 
    end
end

