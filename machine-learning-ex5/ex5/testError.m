function [error_test] = testError(X,y,Xval, yval, Xtest, ytest,lambda) 

theta = [1;1]
m = length(X);

X = [ones(m,1) X];
Xval = [ones(length(Xval),1) Xval];
Xtest = [ones(length(Xtest),1) Xtest];

theta = trainLinearReg(X,y, lambda)
error_val = linearRegCostFunction(Xval, yval, theta, 0)
error_test = linearRegCostFunction(Xtest,ytest, theta, 0)