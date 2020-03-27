% Script to execute Simple Linear Regression
function ex1()
    data1 = load('ex1data1.txt');
    x = data1(:,1);
    y = data1(:,2);

    % plotData(x,y);

    m = size(y,1); % Size of training data
    X = [ones(m,1), x]; % X matrix
    theta = zeros(size(X,2),1); % Initialize theta

    computeCost(X, y, theta);

    iterations = 1500;
    alpha = 0.001;

    [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

    predict = sum(theta'.* X, 2);

    % scatter(linspace(1,iterations,iterations), J_history);

    plot(x, y,'rx', 'MarkerSize',10);
    hold;
    plot(x, predict);

end