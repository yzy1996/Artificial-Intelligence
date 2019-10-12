%% import data
poly_data = load('poly_data.mat');
sampx = poly_data.sampx;
sampy = poly_data.sampy;
polyx = poly_data.polyx;
polyy = poly_data.polyy;

%% show the data plot
hold on
scatter(sampx,sampy,'.')
plot(polyx,polyy,'red')

%% pretreatment
[m, n] = size(sampx);
Phi = ones(6, n);

for i = 1:n
    Phi(:,i) = [1, sampx(i), sampx(i)^2, sampx(i)^3, sampx(i)^4, sampx(i)^5];
end

%% Least-squares
theta1 = (Phi*Phi')\(Phi*sampy);
plot(polyx,polyval(flipud(theta1),polyx),'green*')

%% Regularized LS
lambda = 0.1;
theta2 = (Phi*Phi' + lambda*eye(6))\(Phi*sampy);
plot(polyx,polyval(flipud(theta2),polyx),'cyano')

%% L1-regularized LS
H = [Phi*Phi' -Phi*Phi';-Phi*Phi' Phi*Phi'];
f = lambda*ones(12,1)-[Phi*sampy;-Phi*sampy];
x = quadprog(H,f);
theta3 = x(1:6) - x(7:12);
plot(polyx,polyval(flipud(theta3),polyx),'blue')

%% Robust regression
f = [zeros(1,6) ones(1,n)]';
A = [-Phi' -eye(n);Phi' -eye(n)];
b = [-sampy sampy]';
x = linprog(f,A,b);
theta4 = x(1:6);
plot(polyx,polyval(flipud(theta4),polyx),'black')

%% Bayesian regression
alpha = 0.1;
sigma2 = var(sampy);
Sigma = (1/alpha*eye(6)+1/sigma2*(Phi*Phi'))^(-1);
theta5 = 1/sigma2*Sigma*Phi*sampy;
plot(polyx,polyval(flipud(theta5),polyx),'magenta')

%%
legend('samples','function','least-squares','regularized LS','L1-regularized','robust regression','Bayesian regression')
hold off
