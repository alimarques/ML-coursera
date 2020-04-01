% Script to execute a Logistic Regression with regularization

% Load data
data = load('ex2data2.txt');

x = data(:, 1:2);
y = data(:, 3);

% Plot data
plot = 0;
if (plot == 1)
    plotData(x, y);

    hold on;
    % Labels and Legend
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')

    % Specified in plot order
    legend('y = 1', 'y = 0')
    hold off;

end

% Construct more features
X = mapFeature(x(:,1), x(:,2));

[m, n] = size(X); % Number of training examples and features

% Regularization
% Set regularization parameter lambda
lambda = 10;

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);


