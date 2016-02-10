%this function costFunction computes the cost and gradient for logistic regression
%   costJ = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

function [costJ, gradient] = costFunction(theta, X, Y)
% X = mxn matrix
% theta = nx1 column vector
% y = mx1 column vector

    m = length(X(:,1));
    costJ = 0;
    gradient = zeros(size(theta)); % gradient = nx1 column vector (same size as theta)

    % Compute the costJ of a particular choice of theta

    H = sigmoid(X*theta); % hypothesis = mx1 column vector

    % compute cost costJ
    costJ = (-1/m) * sum(transpose(Y).*log(H) + (1-transpose(Y)).*(1-log(H)));

    % Compute the partial derivatives and set gradient to the partial
    % derivatives of the cost w.r.t. each parameter in theta

    % compute the gradient 
    for i = 1:m
        gradient = gradient + ( H(i) - Y(i) ) * X(i, :)';
    end

    %gradient = (1/m) * gradient;

end