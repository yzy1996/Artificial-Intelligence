function Y = meanshift(x)

iteration = 300;
n = length(x);
h1 = 10;
h2 = 1;
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
end

Y = round(xx(1, :));

% Y(Y < ((max(Y)+min(Y))/2)) = 1;
% Y(Y >= ((max(Y)+min(Y))/2)) = 2;

end