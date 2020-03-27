% Script to execute Multi Linear Regression
function ex1_multi()
    data2 = load('ex1data2.txt');
    x = data2(:,1:2);
    y = data2(:,3);
    m = size(y,1); % Size of training data
    
    [X_norm, mu, sigma] = featureNormalize(x); % Normalize features
  
    X = [ones(m,1), X_norm]; % X matrix
    theta = zeros(size(X,2),1); % Initialize theta
    
    computeCostMulti(X, y, theta);
    
    iterations = 1500;
    alpha = 0.1;

    [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, iterations);
    
    % scatter(linspace(1,iterations,iterations), J_history);
    
    thetaNorm = normalEqn(X,y);
    
  end