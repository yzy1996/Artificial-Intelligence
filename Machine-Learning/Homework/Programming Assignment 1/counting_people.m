clc;clear;
main

function main

count_data = load('count_data.mat');
trainx = count_data.trainx;  %9*400
trainy = count_data.trainy;  %400*1
testx = count_data.testx;
testy = count_data.testy; 

err = zeros(2,5);
Phi = mytrans(trainx);
[m, n] = size(Phi);
K = m-1;

%%
figure
subplot(2,3,1)
hold on
grid on
plot(testy,'.')

theta1 = myls;
plot(mytrans(testx)'*theta1,'.')
title('least-squares')
hold off 
err(1,1) = mymse(theta1);
err(2,1) = mymae(theta1);
%%
subplot(2,3,2)
hold on
grid on
plot(testy,'.')

theta2 = myrls;
plot(mytrans(testx)'*theta2,'.')
title('regularized LS')
hold off 
err(1,2) = mymse(theta2);
err(2,2) = mymae(theta2);
%%
subplot(2,3,3)
hold on
grid on
plot(testy,'.')

theta3 = mylasso;
plot(mytrans(testx)'*theta3,'.')
title('L1-regularized LS')
hold off 
err(1,3) = mymse(theta3);
err(2,3) = mymae(theta3);
%%
subplot(2,3,4)
hold on
grid on
plot(testy,'.')

theta4 = myrr;
plot(mytrans(testx)'*theta4,'.')
title('robust regression')
hold off 
err(1,4) = mymse(theta4);
err(2,4) = mymae(theta4);
%%
subplot(2,3,5)
hold on
grid on
plot(testy,'.')

theta5 = mybr;
plot(mytrans(testx)'*theta5,'.')
title('Bayesian regression')
hold off 
err(1,5) = mymse(theta5);
err(2,5) = mymae(theta5);
%%
method = ["least-squares", "regularized LS", "L1-regularized LS", "robust regression", "Bayesian regression"];
fprintf('the %s algorithm mean-squared error is %f, mean-absolute error is %f\n', [method;err(1,:);err(2,:)])

%% feature transformations
    function Phi = mytrans(x)
%         Phi = x(1,:);Phi = [Phi; Phi.^2];  % 1381
%         Phi = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));Phi = Phi(1,:);
%         Phi = x ./ (1+abs(x));  % 2404
%         Phi = sin(x);  % 2120
%         Phi = atan(x);  % 1866 
%         Phi = x(1,:);  % 1418
        Phi = x;  % 1751
%         Phi = [x; x.^2];  % 1603
%         Phi = [x; x.^2 ;x.^3];  % 1573
%         Phi = 1 ./ (1+ exp(x));  % 4128
%         Phi = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));  % 2153      

    end

%% error
    function err = mymae(theta)       
        err = sum(abs(testy-mytrans(testx)'*theta))/600;
    end

    function err = mymse(theta)
        err = sum((testy-mytrans(testx)'*theta).^2)/600;
    end

%% Least-squares
    function theta1 = myls
        
        theta1 = (Phi*Phi')\(Phi*trainy);
        
    end

%% Regularized LS
    function theta2 = myrls
        
        lambda = 0.1;
        theta2 = (Phi*Phi' + lambda*eye(K+1))\(Phi*trainy);
        
    end

%% L1-regularized LS
    function theta3 = mylasso
        
        lambda = 0.1;
        H = [Phi*Phi' -Phi*Phi';-Phi*Phi' Phi*Phi'];
        H=(H+H')/2;
        f = lambda*ones(2*(K+1),1)-[Phi*trainy;-Phi*trainy];
%         options = optimoptions('quadprog','MaxIter',2000);
        x = quadprog(H,f);
        theta3 = x(1:K+1) - x(K+2:2*K+2);
        
    end

%% Robust regression
    function theta4 = myrr
        
        n = length(trainx);
        f = [zeros(1,K+1) ones(1,n)]';
        A = [-Phi' -eye(n);Phi' -eye(n)];
        b = [-trainy trainy]';
        x = linprog(f,A,b);
        theta4 = x(1:K+1);
        
    end

%% Bayesian regression
    function theta5 = mybr
        
        alpha = 0.1;
        sigma2 = var(trainy);
        Sigma = (1/alpha*eye(K+1)+1/sigma2*(Phi*Phi'))^(-1);
        theta5 = 1/sigma2*Sigma*Phi*trainy;
        
    end
end