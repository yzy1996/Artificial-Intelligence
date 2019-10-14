clc;clear;
main

function main
%% import data
poly_data = load('poly_data.mat');

Sampx = poly_data.sampx;
Sampy = poly_data.sampy;
polyx = poly_data.polyx;
polyy = poly_data.polyy;

%% control training data
% random subset of samples 10%, 25%, 50%, 75%

error = zeros(5,5);
K = 5;
m = length(Sampx);
percent = [0.75, 0.5, 0.25 ,0.1];
iteration = 100;
subtitle = ["75%data", ]
for subset = 1:5
    
    for iter = 1:iteration
        
        suborder = randperm(m, round(m*percent(subset)));
        sampx = Sampx(suborder);
        sampy = Sampy(suborder);
        
        %% pretreatment feature transformation
        n= length(sampx);
        
        Phi = mytrans(sampx);
        
        %% regression algorithms
        
        theta1 = myls;
        error(subset,1) = error(subset,1) + mymse(theta1);
        theta2 = myrls;
        error(subset,2) = error(subset,2) + mymse(theta2);
        theta3 = mylasso;
        error(subset,3) = error(subset,3) + mymse(theta3);
        theta4 = myrr;
        error(subset,4) = error(subset,4) + mymse(theta4);
        [theta5,sigmaa] = mybr;
        error(subset,5) = error(subset,5) + mymse(theta5);
        
    end
    %% show the data plot
    subplot(2,3,subset)
    hold on
    scatter(sampx,sampy,'.')
    plot(polyx,polyy,'red')
    plot(polyx,polyval(flipud(theta1),polyx),'green*')
    plot(polyx,polyval(flipud(theta2),polyx),'cyano')
    plot(polyx,polyval(flipud(theta3),polyx),'blue')
    plot(polyx,polyval(flipud(theta4),polyx),'black')
    plot(polyx,polyval(flipud(theta5),polyx),'magenta')
    
    title('Regression algorithms Plot')
    xlabel('x')
    ylabel('y')
    legend('samples','function','least-squares','regularized LS','L1-regularized','robust regression','Bayesian regression')
    hold off
    
%     figure
%     plot(polyx,sqrt(diag(sigmaa)))
%     title('standard deviation around the mean')
    
end

error = sqrt(error);

subplot(2,3,6)
hold on
plot(percent(1:4),error(1:4,1)/iteration)
plot(percent(1:4),error(1:4,2)/iteration)
plot(percent(1:4),error(1:4,3)/iteration)
plot(percent(1:4),error(1:4,4)/iteration)
plot(percent(1:4),error(1:4,5)/iteration)
legend("least-squares", "regularized LS", "L1-regularized LS", "robust regression", "Bayesian regression")
hold off


method = ["least-squares", "regularized LS", "L1-regularized LS", "robust regression", "Bayesian regression"];
fprintf('the %s algorithm average mean-squared error is %f\n', [method;error(4,:)/iteration])

%% Least-squares
    function theta1 = myls
        
        theta1 = (Phi*Phi')\(Phi*sampy);
        
    end

%% Regularized LS
    function theta2 = myrls
        
        lambda = 0.1;
        theta2 = (Phi*Phi' + lambda*eye(K+1))\(Phi*sampy);
        
    end

%% L1-regularized LS
    function theta3 = mylasso
        
        lambda = 0.1;
        H = [Phi*Phi' -Phi*Phi';-Phi*Phi' Phi*Phi'];
        f = lambda*ones(2*(K+1),1)-[Phi*sampy;-Phi*sampy];
        x = quadprog(H,f);
        theta3 = x(1:K+1) - x(K+2:K+7);
        
    end

%% Robust regression
    function theta4 = myrr
        
        f = [zeros(1,K+1) ones(1,n)]';
        A = [-Phi' -eye(n);Phi' -eye(n)];
        b = [-sampy sampy]';
        x = linprog(f,A,b);
        theta4 = x(1:K+1);
        
    end

%% Bayesian regression
    function [theta5,sigmaa] = mybr
        
        alpha = 0.1;
        sigma2 = var(sampy);
        Sigma = (1/alpha*eye(K+1)+1/sigma2*(Phi*Phi'))^(-1);
        theta5 = 1/sigma2*Sigma*Phi*sampy;
        sigmaa = mytrans(polyx)'*Sigma*mytrans(polyx);
        
    end

%% function
    function err = mymse(theta)
        % mean-squared error
        
        err = sum((polyy'-polyval(flipud(theta),polyx)).^2);
    end

    function Phi = mytrans(sampx)
        n = length(sampx);
        Phi = ones(K+1, n);
        for i = 1:n
            Phi(:,i) = [1, sampx(i), sampx(i)^2, sampx(i)^3, sampx(i)^4, sampx(i)^5];
        end
    end

end


