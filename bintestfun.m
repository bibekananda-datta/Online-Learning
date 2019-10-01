function [NMistake] = bintestfun(xdata,ydata,W)

% bintestfun function is to predict binary classifiers

% Author    : Bibekananda Datta
% Date      : 09/28/2019 

% input arguments/ parameters:
% xdata     : x test data as matrix (feature vectors as column)
% ydata     : y test data as column vector
% W         : learned weight vector

% output arguments/ returning variables:
% NMist     : no of mistakes in prediction

Y           = sign(W*xdata');       %checks the sign of test data
NMistake    = 0;                    %initialize mistake counter

for i = 1:length(Y)             
    if Y(i) ~= ydata(i)
        NMistake = NMistake +1;     %updates counter for each mistake in prediction
    end
end

% this function can be modified to return accuracy instead of no of mistakes
% following segment calculates accuracy
% change output argument to Accuracy from NMistake for this purpose

% Accuracy    = 1-NMistake/datasize;


end