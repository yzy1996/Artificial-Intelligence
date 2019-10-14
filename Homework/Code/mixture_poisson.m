%% Dataset1-London
% x(1:229)=0;
% x(230:440)=1;
% x(441:533)=2;
% x(534:568)=3;
% x(569:575)=4;
% x(576)=5;
% n=size(x,2);

%% Dataset2-Antwerp
x(1:325)=0;
x(326:440)=1;
x(441:507)=2;
x(508:537)=3;
x(538:555)=4;
x(556:576)=5;
n=size(x,2);

%% Initialize
h=animatedline; %画图
K=5;      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%修改这个K
pi=rand(1,K);
pi=pi/sum(pi);
lambda=10*rand(1,K);
iteration=1000;      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%修改这个iteration

%% EM
gamma=zeros(n,K);
poisson=zeros(n,K);
gg=zeros(1,K);

for iter=1:iteration
    
    % E-STEP
    for i=1:n
        for j=1:K
            poisson(i,j)=pi(j)*(1/factorial(x(i)))*exp(-lambda(j))*(lambda(j)^x(i));
        end
    end
    
    for i=1:n
        for j=1:K
            gamma(i,j)=poisson(i,j)/sum(poisson(i,:));
        end
    end
    
    % M-STEP
    for j=1:K
        lambda(j)=sum(gamma(:,j).*x')/sum(gamma(:,j));
        
        gg(j)=sum(gamma(:,j));
    end
    
    for j=1:K
        pi(j)=gg(j)/sum(gg);
    end
    
    addpoints(h,iter,lambda(1));
    drawnow
end

fprintf('the value of lambda is %4.3f ,pi is %4.3f\n',[lambda;pi])