clc;clear;
format long
% dbstop if error

%% input data
cluster_data = load('cluster_data.mat');
x = cluster_data.dataA_X;
y = cluster_data.dataA_Y;
[d, n] = size(x);  % the number of datax
scalemin = min(x, [], 2);  % the min scale of datax
scalemax = max(x, [], 2);  % the max scale of datax
%% initial
K = 4;
iteration = 200;  % fix maximum number of iterations

%% truth ground
subplot(2,2,1)
scatter(x(1, :), x(2, :), 10, y, 'filled');
title('ground truth')

%% K-means algorithm

z = zeros(n, K);
mu = scalemin + (scalemax - scalemin) .* rand(d, K);
dis = zeros(1,K);
look = mu;


for iter = 1:iteration
    for i = 1:n
        for j = 1:K
            dis(j) = sum((x(:, i) - mu(:, j)).^2);
        end
        z(i,:) = dis == min(dis);  % using logical array
    end

    for j = 1:K
        if sum(z(:, j)) ~= 0  % avoid 0 division!!
            mu(:, j) = (x * z(:, j)) / sum(z(:, j));
        end
    end

    % end the iteration early
    if look == mu
        break
    end
    look = mu;
end

y1 = (1:K) * z';

y11 = zeros(1, n);
y11(y1 == y1(find(y == 1,1))) = 1;
y11(y1 == y1(find(y == 2,1))) = 2;
y11(y1 == y1(find(y == 3,1))) = 3;
y11(y1 == y1(find(y == 4,1))) = 4;
error1 = sum(y11 ~= y)

subplot(2,2,2)
scatter(x(1, :), x(2, :), 10, y11, 'filled');
title('K-means')

%% EM-GMM

mu = scalemin + (scalemax - scalemin) .* rand(d, K);

sigma = repmat(eye(d),[1 1 K]);
pi=rand(1,K);
pi=pi/sum(pi);
look = pi;
gmm = zeros(n, d);
z = zeros(n, K);

for iter = 1:iteration
    
    % E-Step
    for i = 1:n
        for j = 1:K
            gmm(i, j) = pi(j) * mvnpdf(x(:, i), mu(:, j), sigma(:,:,j));
        end
    end
    
    for i = 1:n
        for j = 1:K
            z(i, j) = gmm(i, j) / sum(gmm(i, :));
        end
    end
    
    % M-Step
    N = sum(z);  % caculate sum of every column
    pi = N / n;
    
    mu = 1 ./ N .* (x * z);
    
    for j=1:K
        sigma(:, :, j) = 1 / N(j) * (x - mu(:, j)) * diag(z(:, j)) * (x - mu(:, j))';
    end
    
    % end the iteration early
    if round(look, 8) == round(pi, 8)
        break
    end
    look = pi;  
end
[M, y2] = max(z,[],2);
y22 = zeros(1, n);
y22(y2 == y2(find(y == 1,1))) = 1;
y22(y2 == y2(find(y == 2,1))) = 2;
y22(y2 == y2(find(y == 3,1))) = 3;
y22(y2 == y2(find(y == 4,1))) = 4;
error2 = sum(y22 ~= y)
subplot(2,2,3)
scatter(x(1, :), x(2, :), 10, y22, 'filled')
title('EM-GMM')

%% Mean-shift algorithm
h = 1;
xx = x;

for i = 1:n

    look = xx(:,i);
    
    for iter = 1:iteration
        xx(:,i) = x * mvnpdf(x', xx(:,i)', h^2 * eye(d)) / sum(mvnpdf(x', xx(:,i)', h^2 * eye(d)));
        
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
error3 = sum(y33 ~= y)
unique(y3(1,:))
subplot(2,2,4)
scatter(x(1, :), x(2, :), 10, y3, 'filled')
title('Mean-shift')

