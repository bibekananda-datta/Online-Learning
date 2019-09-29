clear;
close;
clc;

tic
disp('test program for multiple classifier algorithms is running...');
disp('loading training and testing data');

%% loading and classifying training data
traindata       = readtable('fashion-mnist_train.csv');     %loading csv file training data
xtr             = traindata(1:end,2:end);                   %retrieving x data from the table
ytr             = traindata(1:end,1);                       %retrieving y data from the table
xtr             = table2array(xtr);                         %converting x data to array
ytr             = table2array(ytr);                         %converting y data to array

%% loading and classifying test data
testdata        = readtable('fashion-mnist_test.csv');      %loading csv file training data
xts             = testdata(1:end,2:end);                    %retrieving x data from the table
yts             = testdata(1:end,1);                        %converting x data to array
xts             = table2array(xts);                         %converting x data to array
yts             = table2array(yts);                         %converting y data to array

disp('NIST data loaded');
%% defining variables
IterMax     = 50;           %max iteration
T           = 1;            %Tau for perceptron
ftsize      = size(xtr,2);  %size of feature vector
trsize      = size(xtr,1);  %training data size
tssize      = size(xts,1);  %test data size
Nclass      = 10;

%% comparing online learning of perceptron and passive-aggresive algorithm
disp('comparing perceptron and passive aggresive algorithm for multi-classifier');
Winit                   = zeros(Nclass,ftsize);    %initializing W for training data
[WPercTr, NmisPercTr]   = multipercept(xtr,ytr,Winit,T,IterMax,Nclass); %calling perceptron function for multi-class
[WPasagTr,NmisPATr]     = multipassagg(xtr,ytr,Winit,IterMax,Nclass);   %calling passive-aggressive function for multiclass

disp('plotting online learning curve');
figure(5);
plot(1:IterMax,NmisPercTr,'bo-',1:IterMax,NmisPATr,'ro-');
title('Online learning curve for multi-classifier training data (Iter #50)');
xlabel('# No of iterations');
ylabel('# No of mistakes');
legend('Perceptron','Passive-Aggressive');


%% Accuracy of perceptron, passive aggressive, average perceptron testing data
disp('comparing accuracy of perceptron, passive-aggressive, and averaged perceptron...');

NIter       = 20;
%initializing weight vectors for all 3 algorithms
WPerc       = zeros(Nclass,ftsize);
WPassAgg    = zeros(Nclass,ftsize);
WAPerc      = zeros(Nclass,ftsize);

%initializing mistake counter for each iteration for all 3 algorithms
NPercTs     = zeros(NIter,1);           
NPaAgTs     = zeros(NIter,1);
NAPercTs    = zeros(NIter,1);

%online training and learning routine
for i = 1:NIter
    [WPerc,NMisPerc]        = multipercept(xtr,ytr,WPerc,T,1,Nclass);
    [WPassAgg,NMisPas]      = multipassagg(xtr,ytr,WPassAgg,1,Nclass);
    [WAPerc,NMisAPerc]      = multiavgpercept(xtr,ytr,WAPerc,T,1,Nclass);
    
    %implementing obtained weight vectors to predict classes of test data
    NPercTs(i)  = multitestfun(xts,yts,WPerc,Nclass);
    NPaAgTs(i)  = multitestfun(xts,yts,WPassAgg,Nclass);
    NAPercTs(i) = multitestfun(xts,yts,WAPerc,Nclass);
    
end

%calculating accuracy of the algorithm from no of mistakes made in each iteration
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