clc; clear;
addpath('Access-MFS')
addpath('Datasets')

% load data
datamatrix = load('langlog.mat');
X = datamatrix.data;
Y = datamatrix.target;

%normalize data
min_val = min(X, [], 1);
range = max(X, [], 1) - min_val;
range(range == 0) = 1;   % 防止除0
X = (X - min_val) ./ range;
X=X';
Y(Y==-1)=0;
Y=Y';

%parameater setting
option.lambda=1;
option.theta=1;
option.mu=1;
option.maxiter=20;
option.k_s=15;
option.k_f=4;
option.stopObj=1e-2;
option.sel_fea=100;
m=0.1;

%semi-supervised setting
number=fix(((1-m)*size(X,2)));
index_u=randperm(size(X,2),number);
index=1:1:size(X,2);
for i=index_u
    index(i)=0;
end
index_l=index(index~=0);
Y_te=Y(index_u,:);
Y(index_u,:)=0;

%Access-MFS 
[selfea_ind,VALUE]= AccessMFS(X,Y,option);

%classification experiment
X_tr=X(:,index_l);
X_te=X(:,index_u);
Y_tr=Y(index_l,:);
Y_trT=Y_tr';
Y_teT=Y_te';
sel_X_tr=X_tr(selfea_ind,:);
sel_X_te=X_te(selfea_ind,:);
sel_X_trT=sel_X_tr';
sel_X_teT=sel_X_te';

[Prior,PriorN,Cond,CondN]=MLKNN_train(sel_X_trT,Y_trT,option.k_s,1);
[~,RL,OE,~,AP,MaF,~,~,~]=MLKNN_test(sel_X_trT,Y_trT,sel_X_teT,Y_teT,option.k_s,Prior,PriorN,Cond,CondN);

%display results
fprintf('RL = %.4f, OE = %.4f\n, AP = %.4f\n, MaF = %.4f\n', RL, OE, AP, MaF);
