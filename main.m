clc; clear;
addpath('Access-MFS')
addpath('Datasets')
addpath('ML-KNN')

% load data
datamatrix = load('langlog.mat');
X = datamatrix.data; 
Y = datamatrix.target;

% normalize data
min_X = min(X, [], 1);
max_min = max(X, [], 1) - min_X;
max_min(max_min == 0) = 1;   
X = (X - min_X) ./ max_min; 
X=X'; %d*n
Y(Y==-1)=0; %0-1 label matrix
Y=Y'; %n*c

% parameter setting
pararange=[1e-3, 1e-2,1e-1,1,10];
option.k_s=15; %number of nearest neighbors for the sample similarity graph
option.k_f=4; %number of nearest neighbors for the label similarity graph
option.stopObj=1e-2; %convergence threshold
option.sel_fea=150; %number of selected features
m=0.4; %percentage of labeled instances
best_RL = 1;
best_OE = 1;
best_AP = 0;
best_MaF = 0;

% parameter tuning
for lambda=pararange
    option.lambda=lambda;
    for theta=pararange
        option.theta=theta;
        for mu=pararange
            option.mu=mu;
            for t=1:5
                % semi-supervised setting
                number=fix(((1-m)*size(X,2)));
                index_u=randperm(size(X,2),number);
                index=1:1:size(X,2);
                for i=index_u
                    index(i)=0;
                end
                index_l=index(index~=0);
                Y_te=Y(index_u,:);
                Y(index_u,:)=0;
                
                % run Access-MFS and select features
                selfea_ind= Access_MFS(X,Y,option); 
                
                % perform ML-KNN classification experiment
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
                [~,RL(1,t),OE(1,t),~,AP(1,t),MaF(1,t),~,~,~]=MLKNN_test(sel_X_trT,Y_trT,sel_X_teT,Y_teT,option.k_s,Prior,PriorN,Cond,CondN);
            end
            mean_RL=mean(RL);
            mean_OE=mean(OE);
            mean_AP=mean(AP);
            mean_MaF=mean(MaF);
            if mean_RL<best_RL
                best_RL=mean_RL;
            end
            if mean_OE<best_OE
                best_OE=mean_OE;
            end
            if mean_AP>best_AP
                best_AP=mean_AP;
            end
            if mean_MaF>best_MaF
                best_MaF=mean_MaF;
            end
        end
    end
end

% display optimal results
fprintf('RL = %.4f, OE = %.4f\n, AP = %.4f\n, MaF = %.4f\n', best_RL, best_OE, best_AP, best_MaF);
