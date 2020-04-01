% Script to execute a Logistic Regression

% Load data
data = load('ex2data1.txt');

x = data(:, 1:2);
y = data(:, 3);

[m, n] = size(x); % Number of training examples and features

% Plot data
plot = 0;
if (plot == 1)
    plotData(x, y);

    hold on;
    % Labels and Legend
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')

    % Specified in plot order
    legend('Admitted', 'Not admitted')
    hold off;

end

% Calculate cost with initial parameters
X = [ones(m,1), x];

initial_theta = zeros(n+1, 1);
[cost, grad] = costFunction(initial_theta, X, y);

% Optimization
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%% Plot Decision Boundary
%plotDecisionBoundary(theta, X, y);

% Predict
predictions = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(predictions == y)) * 100);

