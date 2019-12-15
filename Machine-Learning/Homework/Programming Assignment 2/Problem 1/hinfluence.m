clc;clear;
format long
% dbstop if error

%% input data
cluster_data = load('cluster_data.mat');
x = cluster_data.dataC_X;
y = cluster_data.dataC_Y;
[d, n] = size(x);  % the number of datax

%% initial
K = 4;
iteration = 200;  % fix maximum number of iterations

h = [0.1, 1, 2, 10];
subtitle = ["h = 0.1", "h = 1", "h = 2", "h = 10"];
%% Mean-shift algorithm

for subset = 1:length(h)

    xx = x;
    
    for i = 1:n
        
        look = xx(:,i);
        
        for iter = 1:iteration
            xx(:,i) = x * mvnpdf(x', xx(:,i)', h(subset)^2 * eye(d)) / sum(mvnpdf(x', xx(:,i)', h(subset)^2 * eye(d)));
            
            if round(look, 10) == round(xx(:,i), 10)
                break
            end
            look = xx(:,i);
        end
    end
    y3 = round(xx);
    y3 = y3(1,:);
    y33 = zeros(1, n);
    y33(y3 == y3(find(y == 1,1))) = 1;
    y33(y3 == y3(find(y == 2,1))) = 2;
    y33(y3 == y3(find(y == 3,1))) = 3;
    y33(y3 == y3(find(y == 4,1))) = 4;
    fprintf('h %.1f error is %f\n', h(subset), sum(y33 ~= y))
    unique(y3(1,:));
    
    subplot(2,2,subset)
    scatter(x(1, :), x(2, :), 10, y3, 'filled')
    title(subtitle(subset))
end
