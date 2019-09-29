function [W,Nmist] = binavgpercept(xdata,ydata,W,T,IterMax)

% binavgpercept function is based on averaged perceptron algorithm
% this function can be used to find averaged weight vectors for binary classifiers

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
ftsize      = size(xdata,2);        %size of feature vector
yout        = zeros(datasize,1);    %size of predicted-y
Nmist       = zeros(IterMax,1);     %size of mistake counter for each iteration
Wsum        = zeros(1,ftsize);      %size of Wsum


for i = 1:IterMax                   %run the loop up to max iteration
    count   = 0;                    %reset mistake counter for each iteration
    
    for j = 1:datasize              %run loop for each data
        
        yout(j) = sign(W*xdata(j,:)');                      %predict y

        if yout(j) ~= ydata(j)                              %if mistake in prediction
            count       = count + 1;                        %count of mistakes in that iteration
            W           = W+T*ydata(j)*xdata(j,:);          %updates weight vector, W
            
            Wsum        = Wsum + W;                         %updating cumulative weight vector, Wsum
        end
    end
    Nmist(i) = count;                                       %no of mistake predicting y in one iteration
end

W      = Wsum/sum(Nmist);                                   %averaged weight vector

end

