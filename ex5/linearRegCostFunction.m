function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
    %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
    %regression with multiple variables
    %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
    %   cost of using theta as the parameter for linear regression to fit the 
    %   data points in X and y. Returns the cost in J and the gradient in grad

    % Initialize some useful values
    m = length(y); % number of training examples

    % Cost
    h = sum(theta' .* X, 2);
    error = h - y;
    J = 1/(2*m) * sum(error.^2);

    % Gradient
    grad = 1/m * ((error)' * X)';

    % Regularization
    regCost = lambda/(2*m) * sum(theta(2:end).^2);
    regGrad = lambda/m * theta;

    J = J + regCost;
    grad(2:end) = grad(2:end) + regGrad(2:end); 

end
