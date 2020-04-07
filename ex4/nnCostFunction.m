function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
    %NNCOSTFUNCTION Implements the neural network cost function for a two layer
    %neural network which performs classification
    %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    %   X, y, lambda) computes the cost and gradient of the neural network. The
    %   parameters for the neural network are "unrolled" into the vector
    %   nn_params and need to be converted back into the weight matrices. 
    % 
    %   The returned parameter grad should be a "unrolled" vector of the
    %   partial derivatives of the neural network.
    %

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    % Setup some useful variables
    m = size(X, 1);

    % Setup y's vectors labeled
    Y = zeros(m, num_labels);
    for i = 1:m
        for j = 1:num_labels
            if y(i) == j
                Y(i,j) = 1;
            end
        end
    end

    %%%% Forward propagation %%%%%%%
    a1 = [ones(m,1) X];

    % Hidden layer
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(m,1) a2];

    % Final layer
    z3 = a2 * Theta2';
    h = sigmoid(z3);

    % Cost
    J = 0;
    for k = 1:num_labels
        jPartial = -1/m * [Y(:,k) .* log(h(:,k)) + (1 - Y(:,k)) .* log(1 - h(:,k))];
        J = J + sum(jPartial);
    end

    reg = lambda/(2*m) * ...
            [sum(sum(Theta1(:,2:end) .^2,2)) + sum(sum(Theta2(:,2:end) .^2,2))];
    J = J + reg;

    %%%%%%%  backpropagation %%%%%%%
    delta3 = h - Y;
    delta2 = [(Theta2(:,2:end))' * delta3']' .* sigmoidGradient(z2);

    Theta2_grad = delta3' * a2/m;
    Theta1_grad = delta2' * a1/m;

    % Regularization
    regTheta1 = lambda/m * Theta1;
    regTheta1(:,1) = 0;
    regTheta2 = lambda/m * Theta2;
    regTheta2(:,1) = 0;

    Theta2_grad = Theta2_grad + regTheta2;
    Theta1_grad = Theta1_grad + regTheta1;


    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
