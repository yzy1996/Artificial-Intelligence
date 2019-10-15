clc;clear;
main

function main
%% import data
poly_data = load('poly_data.mat');

Sampx = poly_data.sampx;
Sampy = poly_data.sampy;
polyx = poly_data.polyx;
polyy = poly_data.polyy;

m = length(Sampx);

f3 = figure;
error = zeros(5,5);

%% control training data
% random subset of samples 10%, 25%, 50%, 75%

K = 5;
percent = [1, 0.75, 0.5, 0.25 ,0.1];
iteration = 20;
subtitle = ["75% data", "50% data", "25% data", "10% data"];


for subset = 1:5
	
    for iter = 1:iteration
        
        suborder = randperm(m, round(m*percent(subset)));
        sampx = Sampx(suborder);
        sampy = Sampy(suborder);
        
        Phi = mytrans(sampx);
        
        %% regression algorithms
        
        theta1 = myls;
        error(subset, 1) = error(subset, 1) + mymse(theta1);
        theta2 = myrls;
        error(subset, 2) = error(subset, 2) + mymse(theta2);
        theta3 = mylasso;
        error(subset, 3) = error(subset, 3) + mymse(theta3);
        theta4 = myrr;
        error(subset, 4) = error(subset, 4) + mymse(theta4);
        [theta5,sigmaa] = mybr;
        error(subset, 5) = error(subset, 5) + mymse(theta5);
        
        %%%%% Problem2
        if subset == 1
            figure
            myplot
            title('plot of the estimated function')
            
            figure
            plot(polyx,sqrt(diag(sigmaa)),'LineWidth',2)
            title('standard deviation around the mean')
            break
        end
        
    end
	
    if subset == 1
        continue
    end
    
    %%%%% Problem3

    figure(f3);
    subplot(2, 2, subset-1)
	myplot
    title(subtitle(subset-1))
   
    %% Add outliers output values
    % add large numbers to a few values in sampy
    
    if subset == 4
        err = zeros(1,5);
        Sampy(suborder) = Sampy(suborder) + 50;
        sampx = Sampx;
        sampy = Sampy;
        Phi = mytrans(sampx);
        theta1 = myls;
        err(1) = mymse(theta1);
        theta2 = myrls;
        err(2) = mymse(theta2);
        theta3 = mylasso;
        err(3) = mymse(theta3);
        theta4 = myrr;
        err(4) = mymse(theta4);
        [theta5,~] = mybr;
        err(5) = mymse(theta5);
        figure
        myplot
        title('outliers output values')
    end
end

error = sqrt(error);
error(2:5,1:5) = error(2:5,1:5)/iteration;
myerrorplot

% print MSE
% method = ["least-squares", "regularized LS", "L1-regularized LS", "robust regression", "Bayesian regression"];
% fprintf('the %s algorithm average mean-squared error is %f\n', [method;error(1,:)])
% fprintf('the %s algorithm average mean-squared error is %f\n', [method;error(2,:)])
% fprintf('the %s algorithm average mean-squared error is %f\n', [method;error(3,:)])
% fprintf('the outliers %s algorithm average mean-squared error is %f\n', [method;err])


% plot error responding reduced data
% ignore 10% because of too high error value
    function myerrorplot
        figure
        hold on
        plot(percent(1:4),error(1:4,1),'LineWidth',1)
        plot(percent(1:4),error(1:4,2),'LineWidth',1)
        plot(percent(1:4),error(1:4,3),'--','LineWidth',1)
        plot(percent(1:4),error(1:4,4),'LineWidth',1)
        plot(percent(1:4),error(1:4,5),'LineWidth',1)
        set(gca,'XDir','reverse')
        legend("least-squares", "regularized LS", "L1-regularized LS", "robust regression", "Bayesian regression")
        hold off
    end

	function myplot
		hold on
		scatter(sampx,sampy, 100, '.')
		plot(polyx, polyy, 'red')
		plot(polyx, polyval(flipud(theta1),polyx), 'green', 'LineWidth',1)
		plot(polyx, polyval(flipud(theta2),polyx), 'cyan', 'LineWidth',1)
		plot(polyx, polyval(flipud(theta3),polyx), 'blue--', 'LineWidth',1)
		plot(polyx, polyval(flipud(theta4),polyx), 'black', 'LineWidth',1)
		plot(polyx, polyval(flipud(theta5),polyx), 'magenta', 'LineWidth',1)
		
		xlabel('x')
		ylabel('y')
		legend('samples','function','least-squares','regularized LS','L1-regularized','robust regression','Bayesian regression')
		hold off
	end

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
        theta3 = x(1:K+1) - x(K+2:2*K+2);
        
    end

%% Robust regression
    function theta4 = myrr
        n = length(sampx);
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

%% other function
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


