function Y = meanshift(x)
iteration = 40;
n = length(x);
h1 = 1;
h2 = 2;
xx = x;

for i = 1:n

    look = xx(:,i);
    
    for iter = 1:iteration
        
        k = 1 / (4 * pi^2 * h1^2 * h2^2) * exp(- 1 / (2 * h1^2) * sum((xx(1:2, i) - x(1:2, :)).^2) - 1 / (2 * h2^2) * sum((xx(3:4, i) - x(3:4, :)).^2));
        xx(:,i) = x * k' / sum(k);
        
        if round(look, 10) == round(xx(:,i), 10)
            break
        end
        look = xx(:,i);
    end
    iter
end

xx = round(xx);
Y = xx(4, :);
end