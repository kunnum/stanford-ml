function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
h = sigmoid(z);
theta_reg = theta(2:end,:);
n = length(theta_reg);

normal_cost = sum(-y .* log(h) - (1 - y) .* log(1 - h)) / m;
reg_cost =  (lambda / (2 * m)) * sum(theta_reg .* theta_reg);

J = normal_cost + reg_cost;

for i = 1:size(grad)(1),
    grad(i) = sum((h - y) .* X(:,i)) / m;
    if i > 1,
        grad(i) = grad(i) + (lambda / m) * theta(i);
end






% =============================================================

end
