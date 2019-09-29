function [W,Nmist] = binpassagg(xdata,ydata,W,IterMax)

% Author    : Bibekananda Datta
% Date      : 09/28/2019 

% input arguments/ parameters:
% xdata     : x training data as matrix (feature vectors as column)
% ydata     : y training data as column vector
% W         : initial weight vector
% IterMax   : maximum no of iteration

% output arguments/ returning variables:
% W         : updated weight vectors,
% NMist     : no of mistakes in each iteration as vector

% initializing necessary variables
datasize    = size(xdata,1);
yout        = zeros(datasize,1);
Nmist       = zeros(IterMax,1);

for i = 1:IterMax                                                       %run the loop up to max iteration
    count   = 0;                                                        %reset mistake counter for each iteration
    for j = 1:datasize                                                  %run loop for each data
        
        yout(j) = sign(W*xdata(j,:)');                                  %predict-y

        if yout(j) ~= ydata(j)                                          %if mistake in prediction
            count   = count + 1;                                        %count of mistake in an iteration
            T       = (1-ydata(j)*W*xdata(j,:)')/norm(xdata(j,:))^2;    %updates learning, T
            W       = W+T*ydata(j)*xdata(j,:);                          %updates weight vector, W
        end
        
    end
    Nmist(i) = count;                                                   %no of mistake predicting y in one iteration
end

end