clc
clear all
% load test_label
% load train_label
load r_label
% Test=test_label;
% Train=train_label-1;
Train=r_label-1;
% save Test Test
save Train Train
