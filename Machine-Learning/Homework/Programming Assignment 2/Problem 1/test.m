clc;clear;
x = [90;125];
mu = 125*rand(2, 1)
sigma = 100*eye(2)
mvnpdf(x, mu, sigma)