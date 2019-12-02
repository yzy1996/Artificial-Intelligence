function Y = emgmm(x, K)

[d, n] = size(x);  % the number of datax

mu = [255 * rand(2, K); 512 * rand(2, K)/10];
sigma = repmat(255*eye(d),[1 1 K]); % ≥ı ºªØsigma
pi=rand(1,K);
pi=pi/sum(pi);
look = pi;
gmm = zeros(n, d);
z = zeros(n, K);

iteration = 100;

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
[~, Y] = max(z,[],2);

end