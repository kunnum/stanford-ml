function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% fprintf("size of X is %d x %d\n", size(X))
% fprintf("size of Theta1 is %d x %d\n", size(Theta1))
% fprintf("size of Theta2 is %d x %d\n", size(Theta2))

l2_i = [ones(m, 1) X];
l2_o = sigmoid(l2_i * Theta1');;

% fprintf("size of l2_o is %d x %d\n", size(l2_o))

l3_i = [ones(m, 1) l2_o];
l3_o = sigmoid(l3_i * Theta2');

% fprintf("size of l3_o is %d x %d\n", size(l3_o))

[val, p] =  max(l3_o, [], 2);
% =========================================================================


end
