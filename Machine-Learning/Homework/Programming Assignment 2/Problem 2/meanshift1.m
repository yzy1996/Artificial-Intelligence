function Y = meanshift1(x,h)

iteration = 300;
[d, n] = size(x);

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

Y = round(xx(1, :));

% Y(Y < ((max(Y)+min(Y))/2)) = 1;
% Y(Y >= ((max(Y)+min(Y))/2)) = 2;

end