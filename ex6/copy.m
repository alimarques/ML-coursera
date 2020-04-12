function [C, sigma] = dataset3Params1(X, y, Xval, yval)
    %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
    %where you select the optimal (C, sigma) learning parameters to use for SVM
    %with RBF kernel
    %   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
    %   sigma. You should complete this function to return the optimal C and 
    %   sigma based on a cross-validation set.
    %
    
    C = 1;
    sigma = 0.1;

    step = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    n = length(step);
    
    sigma = step(1);
    C = step(1);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predTest = svmPredict(model, Xval);
    costCompare = mean(double(predTest ~= yval));
    
    for i = 1:n
        newC = step(i);
        for j = 1:n
            newSigma = step(j);
            model = svmTrain(X, y, newC, @(x1, x2) gaussianKernel(x1, x2, newSigma));
            predTest = svmPredict(model, Xval);
            newCost = mean(double(predTest ~= yval));
            
            if(newCost <= costCompare)
                sigma = newSigma;
                C = newC;
                costCompare = newCost;
            end
        end    
    end

end
