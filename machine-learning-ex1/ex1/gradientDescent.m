function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    theta = theta - (alpha / m) * ((theta' * X' - y') * X)';
    
    %       2x1       1       1       1x2  2x97   97x1  97x2
    %temp_1 = theta(1) - alpha / m * sum((theta' * X() - y) * x(:,1)
    %temp_2 = theta(2) - alpha / m * (theta' * X(:,2) - y) * X(:,2)
    %theta(1) = temp_1
    %theta(2) = temp_2
    %size(X)
    %size(theta)
    %size(y)
    %theta
    %X
    %pred = theta' * X'
    %theta = theta - alpha / m * ((X * theta - y)' * X)
    %computeCost(X, y , theta)
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================
    % Save the cost J in every iteration
    %size(computeCost(X, y, theta))    
    J_history(iter) = computeCost(X, y, theta);

end

end
