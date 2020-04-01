function plotData(X, y)
    %PLOTDATA Plots the data points X and y into a new figure 
    %   PLOTDATA(x,y) plots the data points with + for the positive examples
    %   and o for the negative examples. X is assumed to be a Mx2 matrix.

    % Create New Figure
    figure; hold on;

    % Find classes
    negative = find(y==0);
    positive = find(y==1);

    plot(X(positive,1), X(positive,2), 'k+', ... % positive class
        'LineWidth', 2, ...
        'MarkerSize', 5, ...
        'MarkerEdgeColor','b');
        
    plot(X(negative,1), X(negative,2), 'ko', ... % negative class
        'LineWidth', 2, ...
        'MarkerSize', 5, ...
        'MarkerFaceColor','r', ...
        'MarkerEdgeColor','r');

    hold off;

end
