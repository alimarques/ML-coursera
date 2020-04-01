function [J, grad] = costFunctionReg(theta, X, y, lambda)
    %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 

    % Initialize some useful values
    m = length(y); % number of training examples

    h = sigmoid(sum(theta'.* X, 2));
    jPartial = -1/m * (y .* log(h) + (1 - y) .* log(1 - h));
    J = sum(jPartial);
    
    % Regularization
    reg = lambda/(2*m) * sum(theta(2:length(theta)) .^2);
    J = J + reg;
    
    % Gradient
    rate = ((h - y)'* X);
    grad = 1/m * rate';
    
    % Gradient regularization
    regGrad = [0; lambda/m * theta(2:length(theta))];
    grad = grad + regGrad;

end
