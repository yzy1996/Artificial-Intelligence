clc;clear;

poly_data = load('poly_data.mat');
sampx = poly_data.sampx;
sampy = poly_data.sampy;
polyx = poly_data.polyx;
polyy = poly_data.polyy;

K = 5;
n = length(sampx);
% Phi = ones(1, n);
% for k = 1:K
%     Phi = [Phi; sampx.^k];
% end

Phi = ones(K+1, n);
for k = 1:K
    Phi(k+1, :) = sampx.^k;
end