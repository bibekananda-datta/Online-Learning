function [W,Nmist] = binpercept(xdata,ydata,W,T,IterMax)

% binpercept function is based on perceptron algorithm
% this function can be used to find weight vectors for binary classifiers

% Author    : Bibekananda Datta
% Date      : 09/28/2019 

% input arguments/ parameters:
% xdata     : x training data as matrix (feature vectors as column)
% ydata     : y training data as column vector
% W         : initial weight vector
% T         : training rate (remains constant)
% IterMax   : maximum no of iteration

% output arguments/ returning variables:
% W         : updated weight vectors,
% NMist     : no of mistakes in each iteration as vector

% initializing necessary variables
datasize    = size(xdata,1);        %size of training data
yout        = zeros(datasize,1);    %size of predicted-y
Nmist       = zeros(IterMax,1);     %size of mistake counter for each iteration


for i = 1:IterMax                   %run the loop up to max iteration
    count   = 0;                    %reset mistake counter for each iteration
    
    for j = 1:datasize              %run loop for each data set

        yout(j) = sign(W*xdata(j,:)');                      %predict y

        if yout(j) ~= ydata(j)
            count       = count + 1;                        %count of mistakes in an iteration
            W           = W+T*ydata(j)*xdata(j,:);          %updates weight vector, W
        end
    end
    Nmist(i) = count;                                       %no of mistake predicting y in one iteration
end

end

