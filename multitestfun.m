function [NMistake] = multitestfun(xts,yts,W,Nclass)

% bintestfun function is to predict binary classifiers

% Author    : Bibekananda Datta
% Date      : 09/28/2019 

% input arguments/ parameters:
% xdata     : x test data as matrix (feature vectors as column)
% ydata     : y test data as column vector
% W         : learned weight vector
% Nclass    : no of class in test dataset (should be same as training data)

% output arguments/ returning variables:
% NMist     : no of mistakes in prediction


datasize    = size(xts,1);              %size of test data
ftsize      = size(xts,2);              %size of feature vector
Y           = zeros(datasize,1);        %initialzie of predicted Y vector
F           = zeros(Nclass,ftsize);     %initialize augmented feature function
        
for i = 1:datasize                      %loop through all data
    for j = 1:Nclass
        F(j,:)     = xts(i,:);          %create augmented feature function
    end

    [~, index]   = max(dot(W',F'));     %argmax (w.F)
    Y(i)            = index-1;          %adjusts index to class label prediction
end
        


NMistake    = 0;                        %initialize mistake counter

for i = 1:length(Y)
    if Y(i) ~= yts(i)
        NMistake = NMistake +1;         %updates counter for each mistake in prediction
    end
end

% this function can be modified to return accuracy instead of no of mistakes
% following segment calculates accuracy
% change output argument to Accuracy from NMistake for this purpose

% Accuracy    = 1-NMistake/datasize;

end
