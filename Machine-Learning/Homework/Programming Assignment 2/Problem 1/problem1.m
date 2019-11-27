clc;clear;
format long
% dbstop if error

%% input data
cluster_data = load('cluster_data.mat');
x = cluster_data.dataA_X;
y = cluster_data.dataA_Y;
[d, n] = size(x);  % the number of datax
scale1 = min(min(x));  % the min scale of datax
scale2 = max(max(x));  % the max scale of datax

%% initial
K = 4;
iteration = 100;  % fix maximum number of iterations

%% K-means algorithm
z = zeros(n, K);
mu = scale1 + (scale2 - scale1) * rand(d, K);
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
yy = (1:K) * z';
figure()
scatter(x(1, :), x(2, :), 36, yy, 'filled');

%% EM-GMM
mu = scale1 + (scale2 - scale1) * rand(d, K);
[sigma(:,:,1),sigma(:,:,2),sigma(:,:,3),sigma(:,:,4)] = deal(eye(d));
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
[M, yy] = max(z,[],2);
figure()
scatter(x(1, :), x(2, :), 36, yy, 'filled')

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

xx = round(xx);
unique(xx(1,:))
figure()
scatter(x(1, :), x(2, :), 36, sum(xx), 'filled')

