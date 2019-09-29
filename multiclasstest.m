clear;
close;
clc;

tic
disp('test program for multiple classifier algorithms is running...');
disp('loading training and testing data');

%% loading and classifying training data
traindata       = readtable('fashion-mnist_train.csv');
xtr             = traindata(1:end,2:end);
ytr             = traindata(1:end,1);
xtr             = table2array(xtr);
ytr             = table2array(ytr);

%% loading and classifying test data
testdata        = readtable('fashion-mnist_test.csv');
xts             = testdata(1:end,2:end);
yts             = testdata(1:end,1);
xts             = table2array(xts);
yts             = table2array(yts);

disp('NIST data loaded');
%% defining variables
NIter       = 50;           %max iteration
T           = 1;            %Tau for perceptron
ftsize      = size(xtr,2);  %size of feature vector
trsize      = size(xtr,1);  %training data size
tssize      = size(xts,1);  %test data size
Nclass      = 10;

%% comparing online learning of perceptron and passive-aggresive algorithm
disp('comparing perceptron and passive aggresive algorithm for multi-classifier');
Winit                   = zeros(Nclass,ftsize);    %defining W for training data
[WPercTr, NmisPercTr]   = multipercept(xtr,ytr,Winit,T,IterMax,Nclass);
[WPasagTr,NmisPATr]     = multipassagg(xtr,ytr,Winit,IterMax,Nclass);

disp('plotting online learning curve');
figure(5);
plot(1:NIter,NmisPercTr,'bo-',1:NIter,NmisPATr,'ro-');
title('Online learning curve for multi-classifier training data (Iter #50)');
xlabel('# No of iterations');
ylabel('# No of mistakes');
legend('Perceptron','Passive-Aggressive');


%% Accuracy of perceptron, passive aggressive, average perceptron testing data
disp('comparing accuracy of perceptron, passive-aggressive, and averaged perceptron...');

NIter       = 20;

WPerc       = zeros(Nclass,ftsize);
WPassAgg    = zeros(Nclass,ftsize);
WAPerc      = zeros(Nclass,ftsize);

NPercTs     = zeros(NIter,1);
NPaAgTs     = zeros(NIter,1);
NAPercTs    = zeros(NIter,1);


for i = 1:NIter
    [WPerc,NMisPerc]        = multipercept(xtr,ytr,WPerc,T,1,Nclass);
    [WPassAgg,NMisPas]      = multipassagg(xtr,ytr,WPassAgg,1,Nclass);
    [WAPerc,NMisAPerc]      = multiavgpercept(xtr,ytr,WAPerc,T,1,Nclass);
    
    NPercTs(i)  = multitestfun(xts,yts,WPerc,Nclass);
    NPaAgTs(i)  = multitestfun(xts,yts,WPassAgg,Nclass);
    NAPercTs(i) = multitestfun(xts,yts,WAPerc,Nclass);
    
end

AccPercTs       = 1-NPercTs/tssize;
AccPaAgTs       = 1-NPaAgTs/tssize;
AccAPercTs      = 1-NAPercTs/tssize;

disp('plotting comparison among the algorithms for binary classifiers');
figure(2);
plot(1:NIter,AccPercTs,'ro-', 1:NIter,AccPaAgTs,'b*-',1:NIter,AccAPercTs,'gd-');
title('Accuracy curve of multi-classifiers on testing data');
xlabel('# No of iterations');
ylabel('Accuracy');
legend('Perceptron','Passive-Aggressive','Average Perceptron');

disp('test program run is complete');
toc