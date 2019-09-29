function [W,Nmist] = multipassagg(xdata,ydata,W,IterMax,Nclass)

% multipassagg function is based on passive-aggressive algorithm for multiple classifiers
% this function can be used to find weight vectors for multiple classifiers

% Author    : Bibekananda Datta
% Date      : 09/28/2019 

% input arguments/ parameters:
% xdata     : x training data as matrix (feature vectors as column)
% ydata     : y training data as column vector
% W         : initial weight vector
% IterMax   : maximum no of iteration
% Nclass    : no of class in training dataset

% output arguments/ returning variables:
% W         : updated weight vectors,
% NMist     : no of mistakes in each iteration as vector

% initializing necessary variables
datasize    = size(xdata,1);            %size of training data
ftsize      = size(xdata,2);            %size of feature vector
yout        = zeros(datasize,1);        %size of predicted-y
Nmist       = zeros(IterMax,1);         %size of mistake counter for each iteration


for i = 1:IterMax                       %run the loop up to max iteration
    count   = 0;                        %reset mistake counter for each iteration
    
    for j = 1:datasize                      %run loop for each data
        
        F      = zeros(Nclass,ftsize);      %define F(xt,yt)
        
        for k = 1:Nclass
            F(k,:)     = xdata(j,:);        %populating F(xt,yt) for y = 1,2,3, .. k
        end
        
        wF              = dot(W',F');
        [maxm, index]   = max(wF);
        %NIST did the class numbering from 0 to 9 whereas MATLAB does starts indexing from 1
        yout(j)         = index-1;                      %adjusting the index to calculate y-predict
        

        if yout(j) ~= ydata(j)                          %if y-predict ~= yt
            count               = count + 1;            %counts no of mistake in an iteration
            F1                  = zeros(Nclass,ftsize); %defines F1 for F(xt,yt)
            F2                  = zeros(Nclass,ftsize); %defines F2 for F(xt,yout)
            F1(ydata(j)+1,:)    = xdata(j,:);           %populating F1 with adjusted index
            F2(yout(j)+1,:)     = xdata(j,:);           %populating F2 with adjusted index
            
            %passive-aggressive algorithm needs training rate, T, to be updated for each mistake
            T                   = (1-(sum(dot(W',F1'))-sum(dot(W',F2'))))/(norm(F1-F2))^2;
            W                   = W + T*(F1-F2);        %updates weight vector, W
        end
       
    end
    Nmist(i) = count;                                   %no of mistake in each iteration

end

end



