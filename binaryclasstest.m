clear;
close;
clc;

tic

disp('test program for binary classifier algorithms is running...');
disp('loading training and testing data');

%% loading and classifying training data
traindata       = readtable('fashion-mnist_train.csv');     %loading csv file training data
xtr             = traindata(1:end,2:end);                   %retrieving x data from the table
ytr             = traindata(1:end,1);                       %retrieving y data from the table
xtr             = table2array(xtr);                         %converting x data to array
ytr             = table2array(ytr);                         %converting y data to array

%given data has 10 classes. to implement and test binary classifier algorithm
%even and odd classes are grouped seperately to obtain 2 different classes
for i = 1:length(ytr)
    if mod(ytr(i),2) == 0
        ytr(i)  = 1;        %even number classifier label 
    else
        ytr(i)  = -1;       %odd number classifier label
    end
end

%% loading and classifying test data
testdata        = readtable('fashion-mnist_test.csv');      %loading csv file training data
xts             = testdata(1:end,2:end);                    %retrieving x data from the table
yts             = testdata(1:end,1);                        %converting x data to array
xts             = table2array(xts);                         %converting x data to array
yts             = table2array(yts);                         %converting y data to array

%given data has 10 classes. to implement and test binary classifier algorithm
%even and odd classes are grouped seperately to obtain 2 different classes
for i = 1:length(yts)
    if mod(yts(i),2) == 0
        yts(i)  = 1;        %even number classifier label 
    else
        yts(i)  = -1;       %odd number classifier label
    end
end

disp('NIST data loaded');

%% variable initilization
IterMax     = 50;               %no of iteration
T           = 1;                %Tau for perceptron
ftsize      = size(xtr,2);      %size of feature vector
trsize      = size(xtr,1);      %training data size
tssize      = size(xts,1);      %test data size

%% comparing online learning of perceptron and passive-aggresive algorithm
disp('comparing perceptron and passive aggresive algorithm for binary classifier...');
Winit                   = zeros(1,ftsize);    %defining initial W for training data
[WPercTr, NmisPercTr]   = binpercept(xtr,ytr,Winit,T,IterMax);
[WPasagTr,NmisPATr]     = binpassagg(xtr,ytr,Winit,IterMax);

disp('plotting online learning curve');
figure(1);
plot(1:IterMax,NmisPercTr,'bo-',1:IterMax,NmisPATr,'ro-');
title('Online learning curve for binary class training data');
xlabel('# No of iterations');
ylabel('# No of mistakes');
legend('Perceptron','Passive-Aggressive');


%% Accuracy of perceptron, passive aggressive, average perceptron on testing data
disp('comparing accuracy of perceptron, passive-aggressive, and averaged perceptron...');

NIter       = 20;

WPerc       = zeros(1,ftsize);
WPassAgg    = zeros(1,ftsize);
WAPerc      = zeros(1,ftsize);

NPercTs     = zeros(NIter,1);
NPaAgTs     = zeros(NIter,1);
NAPercTs    = zeros(NIter,1);

for i = 1:NIter
    
    [WPerc,NMisPerc]    = binpercept(xtr,ytr,WPerc,T,1);
    [WPassAgg,NMisPas]  = binpassagg(xtr,ytr,WPassAgg,1);
    [WAPerc,NMisAPerc]  = binavgpercept(xtr,ytr,WAPerc,T,1);
    
    NPercTs(i)          = bintestfun(xts,yts,WPerc);
    NPaAgTs(i)          = bintestfun(xts,yts,WPassAgg);
    NAPercTs(i)         = bintestfun(xts,yts,WAPerc);
    
end

AccPercTs       = 1-NPercTs/tssize;
AccPaAgTs       = 1-NPaAgTs/tssize;
AccAPercTs      = 1-NAPercTs/tssize;

disp('plotting comparison among the algorithms for binary classifiers');
figure(2);
plot(1:NIter,AccPercTs,'ro-', 1:NIter,AccPaAgTs,'b*-',1:NIter,AccAPercTs,'gd-');
title('Accuracy curve for binary classifiers on testing data');
xlabel('# No of iterations');
ylabel('Accuracy');
legend('Perceptron','Passive-Aggressive','Average Perceptron');

disp('test program run is complete');
toc