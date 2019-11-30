clc;clear;

%% import data

poly_data = load('poly_data.mat');
sampx = poly_data.sampx;
sampy = poly_data.sampy;
polyx = poly_data.polyx;
polyy = poly_data.polyy;

%% feature transformation

K = 5;
n = length(sampx);
Phi = ones(K+1, n);
for i = 1:n
    Phi(:,i) = [1, sampx(i), sampx(i)^2, sampx(i)^3, sampx(i)^4, sampx(i)^5];
end

%% regression algorithms

err = zeros(1,5);
theta1 = myls(Phi,sampy);
err(1) = mymse(theta1,polyy,polyx);
theta2 = myrls(Phi,K,sampy);
err(2) = mymse(theta2,polyy,polyx);
theta3 = mylasso(Phi,K,sampy);
err(3) = mymse(theta3,polyy,polyx);
theta4 = myrr(Phi,K,sampx,sampy);
err(4) = mymse(theta4,polyy,polyx);
theta5 = mybr(Phi,K,sampy);
err(5) = mymse(theta5,polyy,polyx);

%% plot

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

%% print MSE

method = ["least-squares", "regularized LS", "L1-regularized LS", "robust regression", "Bayesian regression"];
fprintf('the %s algorithm average mean-squared error is %f\n', [method;err])


%% Least-squares
function theta1 = myls(Phi,sampy)

theta1 = (Phi*Phi')\(Phi*sampy);

end

%% Regularized LS
function theta2 = myrls(Phi,K,sampy)

lambda = 0.1;
theta2 = (Phi*Phi' + lambda*eye(K+1))\(Phi*sampy);

end

%% L1-regularized LS
function theta3 = mylasso(Phi,K,sampy)

lambda = 0.1;
H = [Phi*Phi' -Phi*Phi';-Phi*Phi' Phi*Phi'];
f = lambda*ones(2*(K+1),1)-[Phi*sampy;-Phi*sampy];
x = quadprog(H,f);
theta3 = x(1:K+1) - x(K+2:2*K+2);

end

%% Robust regression
function theta4 = myrr(Phi,K,sampx,sampy)

n = length(sampx);
f = [zeros(1,K+1) ones(1,n)]';
A = [-Phi' -eye(n);Phi' -eye(n)];
b = [-sampy sampy]';
x = linprog(f,A,b);
theta4 = x(1:K+1);

end

%% Bayesian regression
function theta5 = mybr(Phi,K,sampy)

alpha = 0.1;
sigma2 = var(sampy);
Sigma = (1/alpha*eye(K+1)+1/sigma2*(Phi*Phi'))^(-1);
theta5 = 1/sigma2*Sigma*Phi*sampy;

end

%% other function
function err = mymse(theta,polyy,polyx)
% mean-squared error
err = sum((polyy'-polyval(flipud(theta),polyx)).^2);
end






