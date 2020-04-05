function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(m,1) X];

% Hidden layer
z2 = Theta1 * X';
a2 = sigmoid(z2);
a2 = [ones(1,m); a2];

% Final layer
z3 = Theta2 * a2;
a3 = sigmoid(z3);

[prob, p] = max(a3);

p = p';

end
