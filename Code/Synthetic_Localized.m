addpath(genpath('./MALSAR-master/MALSAR/functions/')); % load function
addpath('./MALSAR-master/MALSAR/utils/'); % load utilities
addpath('./functions/'); % load utilities
addpath('./cvx/'); % load utilities
rng(2);
r = 10;% # region
n = 30;% # sample size in each region
dl = 10; % # local feature
dr = 8; % # region feature for visualization

coord = cell(1,r);%spatial coordinates
center = [1,5; % spatial coordinates center
        0,-4; 
        3,1;
        4,5;
        4,-3;
        -1,2;
        -2,3;
        -3,-1;
        -4,-2;
        -5,-5
        ];
    
std = 1;
for i = 1:r
    coord{i}(:,1) = normrnd(center(i,1),std, [n,1]);
    coord{i}(:,2) = normrnd(center(i,2),std, [n,1]);
end
%generate local variable

X_L = cell(1,r);
for i = 1:r
    X_L{i} = rand(n,dl); 
end
%generate regional variable
X_R_local = cell(1,r);
X_R = zeros(r,dr);

%generate regional variable using coordinates information
rng('shuffle');

for i = 1:r
     X_R_local{i}(:,1) = 1*(coord{i}(:,1)) + randn(n,1)./10;
     X_R(i,1) = mean(X_R_local{i}(:,1));
        
     X_R_local{i}(:,2) = -0.1*(coord{i}(:,2)) + randn(n,1)./10;
     X_R(i,2) = mean(X_R_local{i}(:,2));
%      
     X_R_local{i}(:,3) = 0.001*(coord{i}(:,2)).*(coord{i}(:,1)) + randn(n,1)./10;
     X_R(i,3) = mean(X_R_local{i}(:,3));
     
     X_R_local{i}(:,4) = -1*(coord{i}(:,2)).^2.*((coord{i}(:,1))) + randn(n,1)./10;
     X_R(i,4) = mean(X_R_local{i}(:,4));
     
     X_R_local{i}(:,5) = 1*(abs(coord{i}(:,2))).*(abs(coord{i}(:,1))) + randn(n,1)./10;
     X_R(i,5) = mean(X_R_local{i}(:,5));
     
     X_R_local{i}(:,6) = -1*(coord{i}(:,2)).^2 - (coord{i}(:,1)).^2  + randn(n,1)./10;
     X_R(i,6) = mean(X_R_local{i}(:,6));
     
     
     X_R_local{i}(:,7) = 1*abs((coord{i}(:,1)))  + randn(n,1)./10;
     X_R(i,7) = mean(X_R_local{i}(:,7));
     
     X_R_local{i}(:,8) = -1*abs((coord{i}(:,2))) + randn(n,1)./10;
     X_R(i,8) = mean(X_R_local{i}(:,8));
end
latent = 1;
tmp = cat(1,X_R_local{:});
X_R_local = cell(1,r);
X_R = zeros(r,latent);
[~,SCORE] = princomp(tmp);
tmp = SCORE(:,1:latent);
ind = 1;
for i = 1:r
     X_R_local{i}(:,1) = tmp(ind:ind+n-1,1)+ randn(n,1)./100;
     X_R(i,1) = mean(tmp(ind:ind+n-1,1));
%         
%      X_R_local{i}(:,2) = tmp(ind:ind+n-1,2)+ randn(n,1)./100;
%      X_R(i,2) = mean(tmp(ind:ind+n-1,2));
     ind = ind+n;
end
dr =latent;
%normalize the data
clear std ind;
tmp = cat(1,X_R_local{:});
tmp = (tmp - repmat(mean(tmp),[n*r,1]))./repmat(std(tmp),[n*r,1]);
tmp1 = cat(1,X_L{:});
tmp1 = (tmp1 - repmat(mean(tmp1),[n*r,1]))./repmat(std(tmp1),[n*r,1]);

ind = 1;
X_R = zscore(X_R);
for i = 1:r
    X_R_local{i} = tmp(ind:ind+n-1,:);
    X_L{i} = tmp1(ind:ind+n-1,:);
    ind = ind + n;
end
clear ind;
%generate cross term
X_csi = createX_csi_local(X_L,X_R_local);

X_csi_R = createX_csi(X_L,X_R);
%local coefficient,region coefficient and vectorized CSI
rng(3);
alphatrue = randn(dl,r);
for i = 1:r
    alphatrue(:,i) = alphatrue(:,i)+ 0.01*i;
end
% alphatrue = normrnd(0,3,[dl,r]);
betatrue = randn(dr,r);
for i = 1:r
    betatrue(:,i) = betatrue(:,i)+ 0.1*i;
end
gammatrue = randn(dl*dr,1);
gammatrue = (abs(gammatrue) > 0.5).*gammatrue.*2;

% generate response variable

Y = cell(1,r);
rng('shuffle') 
for i = 1:r
    Y{i} = X_L{i}*alphatrue(:,i) + (X_R_local{i}*betatrue(:,i)) + X_csi{i}*gammatrue +  randn(n,1);
%     Y{i} = X_L{i}*alphatrue(:,i) + (repmat(X_R(i,:),[size(X_L{i},1),1])*betatrue(:,i)) + X_csi_R{i}*gammatrue +  randn(n,1);
end
Gtrue = reshape(gammatrue,[dr,dl])';
tic;
varname = 'Syn';
% generate trn tst index ------------------------------------
%trnrate = 0.5;
% the number of tasks

crossvalind('kfold',size(Y{1},1),10);%10 fold
Xtrn = cell(1,r);
Ytrn = cell(1,r);
Xtst = cell(1,r);
Ytst = cell(1,r);
coordtrn = cell(1,r);
coordtst = cell(1,r);


cvind = cell(1,size(X_L,2));
rng;
for i = 1:size(X_L,2)
   cvind{i} =  crossvalind('kfold',size(Y{i},1),10);
end

d = size(Xtrn{1},2)-1;
dcsi = size(X_R,2);

rng(2);

Gcli = zeros(dl,dr);
G_estimateint = zeros(dl,dr);
G_dirty = zeros(dl,dr);
G_Leastlasso = zeros(dl,dr);

F1_imput = zeros(10,1);
F1_CLI = zeros(10,1);
F1_MTMLa = zeros(10,1);
Sign_imput = zeros(10,1);
Sign_CLI = zeros(10,1);
Sign_MTMLa = zeros(10,1);
%%  ========================================================================
%%  Cross Validation Split data                                                          
%%  ========================================================================
for ROUND =1 :10 
 
tic;
% Load data
clearvars -except F1_imput X_csi_R X_csi dl dr F1_CLI F1_MTMLa Sign_imput Sign_CLI Sign_MTMLa Gcli X_R_local cvind coord G_Leastlasso G_estimateint G_dirty ROUND alphatrue betatrue gammatrue X_csi_syn X_L X_R Y cvind Xtrn Xtst Ytrn Ytst r d dl dr dcsi varname Wtrue Gtrue;
% load 'D:\sy\4MultiLevel\data\data1.mat';
testIdx = cell(1,r);
trainIdx = cell(1,r);
for i = 1:r
    testIdx{i} = find(cvind{i}==ROUND);
    trainIdx{i} = find(cvind{i}~=ROUND);
    Xtrn{i} = X_L{i}(trainIdx{i},:);
    Xtst{i} = X_L{i}(testIdx{i},:);
    Ytrn{i} = Y{i}(trainIdx{i},:);
    Ytst{i} = Y{i}(testIdx{i},:);
    coordtrn{i} = coord{i}(trainIdx{i},:);
    coordtst{i} = coord{i}(testIdx{i},:);
end


Trnsize = 0;
Tstsize = 0;
for i = 1:r
    Trnsize = Trnsize + size(Ytrn{i},1);
    Tstsize = Tstsize + size(Ytst{i},1);
end


%concatenate local and regional variable
Xtr_LR = cell(r,1);
for i = 1:r
   Xtr_LR{i} = horzcat(Xtrn{i},repmat(X_R(i,:),size(Xtrn{i},1),1));
end

Xtst_LR = cell(r,1);
for i = 1:r
   Xtst_LR{i} = horzcat(Xtst{i},repmat(X_R(i,:),size(Xtst{i},1),1)); 
end



%get the global dataframe
XtrnG = cat(1,Xtr_LR{:});
YtrnG = cat(1,Ytrn{:});
XtstG = cat(1,Xtst_LR{:});
YtstG = cat(1,Ytst{:});

%get the region idx
Rtrnidx = zeros(size(XtrnG,1),1);
Rtstidx = zeros(size(XtstG,1),1);
samplesize_tst = zeros(r,1);

Xtrncsi = zeros(size(XtrnG,1),dcsi);
Xtstcsi = zeros(size(XtstG,1),dcsi);

ind = 1;
for i = 1:dl
    for j = dl+1:dl+dr
        Xtrncsi(:,ind) = XtrnG(:,i).*XtrnG(:,j);
        Xtstcsi(:,ind) = XtstG(:,i).*XtstG(:,j);
        ind = ind + 1;
    end
end

for i = 1:r
    samplesize_tst(i) = size(Xtst{i},1);
end

regiontst = createregion(samplesize_tst);


%validation rate
vadrate = 0.1;




%%  ========================================================================
%%  Initialization                                                          
%%  ========================================================================
result = cell(0);result{1,1} = 'obj func'; result{1,2} = 'runtime'; result{1,3} = 'rmse';result{1,4} = 'r2';
result{1,5} = 'rmse per region';result{1,6} = 'r2 per region';result{1,7} = 'ypred';result{1,8} = 'yreal';
result{1,9} = 'model w';result{1,10} = 'best param';result{1,11} = 'perform_mat';result{1,12} = 'funcFval';result{1,13} = 'model G';
higher_better = false;  % rmse is lower the better.
% param_range = [0.0001:0.0001:0.001,0.002:0.001:0.01,0.2:0.1:1];
param_range = [0.01,0.1,0.5,1];
% param_range = [.1,.001];

% optimization options
opts.init = 2;      % guess start point from data.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance.
opts.maxIter = 500; % maximum iteration number of optimization.
opts.verbose = 0;
opts.OutermaxIter = 100;



% [Xtrn0, Ytrn0, Xtst0, Ytst0] = SplitTrnTst4(data_stl,Trnidx ,Tstidx);
%% Imputation
tic;
method = 1;
% [Wl,Wr,Gestimate,Xlr,Xlrtst,fval] = MTML_imputation(Ytrn,Xtrn,Xtst,X_R,0,0.35,0.2,5,coord,randn(dl,dr),trainIdx,testIdx,50,1e-3);
% lambda1_range = [0];
% lambda2_range = [0.35];
% lambda3_range = [0.02];
% lambda4_range = [10];
lambda1_range = [0];
% lambda2_range = [0.8];%30 size
lambda2_range = [0.8];
lambda3_range = [0.02];
%lambda3_range = [0.02];
lambda4_range = [10];
param_set = generateParamset4(lambda1_range,lambda2_range,lambda3_range,lambda4_range);
[bestpara_ind,perform_mat] = TuneParam_impute(param_set,Xtrn,Ytrn,X_R,coordtrn,trainIdx,testIdx,vadrate);
Ypredtst = cell(1,r);
bestpara = param_set(bestpara_ind,:);
[Wl,Wr,Gestimate,Xlr,Xlrtst,fval] = MTML_imputation(Ytrn,Xtrn,Xtst,X_R,bestpara(1),bestpara(2),bestpara(3),bestpara(4),coord,randn(dl,dr),trainIdx,testIdx,50,1e-5);
[cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,Gestimate);
[accuracysign,cmsign] = CalcSign(Gestimate,Gtrue);
% two stage
Xlrtrn = cell(1,r);
for i = 1:r
    Xlrtrn{i} = Xlr{i}(trainIdx{i},:);
end
xall = cell(1,r);
GestimateBi = double(Gestimate~=0);
gammabi = reshape(GestimateBi',[dl*dr,1]);
indnzero = find(gammabi~=0);

[Wl,gamma] =  cv_solver(Ytrn,Xtrn,Xlrtrn,1,0,Gestimate,500,1e-11);
gammamat = repmat(gamma,[1,r]);
W = vertcat(Wl,gammamat);

Xreg_tst = cell(1,r);
for i = 1:r
    tmp = Xtst_LR{i};
    tmp(:,dl+1:end) = tmp(:,dl+1:end);
    Xreg_tst{i} = tmp(:,dl+1:end);
end
Xcsitst = createX_csi_local(Xtst,Xreg_tst);
Xtst_noise = cell(1,r);
for i = 1:r
    Xcsitst{i} = Xcsitst{i}(:,indnzero);
end
for i = 1:r
    Xtst_noise{i} = horzcat(Xtst{i},Xlrtst{i},Xcsitst{i});
end
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst_noise,Ytst,W);
% norm(cat(1,Xlr{:})-cat(1,X_R_local{:}))
result{method+1,1} = 'MTML_imputation'; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = yreal;result{method+1,8} = ypred;result{method+1,9} = Wl;result{method+1,10} = bestpara;result{method+1,11} = perform_mat;result{method+1,13} = Gestimate;result{method+1,14} = cm;result{method+1,15} = [cmsign];result{method+1,16} = [accuracysign];result{method+1,17} = [F1];
clear param_set;

%%  One global model (lasso)
tic;
method = method + 1;
obj_func_str = 'STL-global model';
eval_func_str = 'eval_rmse';
% param_range = 0.01*(1:10);
[best_param, perform_mat] = TuneParam_lasso2(param_range,XtrnG, YtrnG,vadrate);
w = lasso(XtrnG,YtrnG,'lambda',best_param);
ypred = XtstG*w;
[r2,rmse] = rsquare(YtstG,ypred);
% rmse = norm(YtstG-ypred)/length(ypred);
rmseall= zeros(r,1); r2all = zeros(r,1);
for t = 1: r
    y_pred_t = ypred(regiontst == t);
    y_t = YtstG(regiontst == t);
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
    
end
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = YtstG;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;
clear ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w t;

%% One global model (lasso) without Region info
tic;
method = method + 1;
obj_func_str = 'STL-global model without Region';
eval_func_str = 'eval_rmse';
% param_range = 0.01*(1:10);
[best_param, perform_mat] = TuneParam_lasso2(param_range,XtrnG(:,1:dl), YtrnG,vadrate);
w = lasso(XtrnG(:,1:dl),YtrnG,'lambda',best_param);
ypred = XtstG(:,1:dl)*w;
[r2,rmse] = rsquare(YtstG,ypred);
% rmse = norm(YtstG-ypred)/length(ypred);
rmseall= zeros(r,1); r2all = zeros(r,1);
for t = 1: r
    y_pred_t = ypred(regiontst == t);
    y_t = YtstG(regiontst == t);
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
end

result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = YtstG;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;
clear ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w t;

%% Cross level interaction
tic;
method = method + 1;
obj_func_str = 'Cross level interaction';
eval_func_str = 'eval_rmse';
% param_range = 0.01*(1:10);
Xtrn_csi = horzcat(XtrnG,Xtrncsi);
Xtst_csi = horzcat(XtstG,Xtstcsi);
[best_param, perform_mat] = TuneParam_lasso2(param_range,Xtrn_csi, YtrnG,vadrate);
w = lasso(Xtrn_csi,YtrnG,'lambda',best_param);
ypred = Xtst_csi*w;
[r2,rmse] = rsquare(YtstG,ypred);
% rmse = norm(YtstG-ypred)/length(ypred);
G = w(dl+dr+1:end);
G = reshape(G,[dr,dl])';
[cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,G);
[accuracysign,cmsign] = CalcSign(G,Gtrue);
Gcli = Gcli + G;

error_CSI = norm(G-reshape(gammatrue,[dl,dr]));
rmseall= zeros(r,1); r2all = zeros(r,1);
for t = 1: r
    y_pred_t = ypred(regiontst == t);
    y_t = YtstG(regiontst == t);
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
end
[cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,Gcli);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = YtstG;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,13} = G;result{method+1,14} = cm;result{method+1,15} = cmsign;result{method+1,16} = [accuracysign];result{method+1,17} = [F1];
clear ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w t G;
%% STL - Lasso (N - Independent model)
tic;
method = method+1;
obj_func_str = 'STL_lasso';
eval_func_str = 'eval_rmse';
% param_range = 10*(1:10);
[best_param] = TuneParam2...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn,Ytrn,vadrate);
% build model using the optimal parameter
W = STL_lasso_trn(Xtrn, Ytrn, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
cvx_begin quiet
variable G(dl, dr+1)
X_z = vertcat(X_R',ones(1,r));
minimize(norm((W - G*X_z),'fro') + 0.1*l1_mat(G))
cvx_end
G = G(:,1:dr);
G = (abs(G)>0.00001).*G;
[cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,G);
[accuracysign,cmsign] = CalcSign(G,Gtrue);
% error_CSI = norm(G(:,1:dr)-reshape(gammatrue,[dr,dl])');
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = 'perform_mat';result{method+1,13} = 'G';result{method+1,16} = [accuracysign];result{method+1,17} = [F1];
result{method+1,15} = cmsign;
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W G;
%% MTL - L21
tic;
method = method+1;
obj_func_str = 'Least_L21';
eval_func_str = 'eval_rmse';
param_range = [0,0.01,0.1,1:10,20:20:100];
[best_param, perform_mat] = TuneParam21...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn,Ytrn,vadrate);
% build model using the optimal parameter
[W,Fval] = Least_L21(Xtrn, Ytrn, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
cvx_begin quiet
variable G(dl, dr+1)
X_z = vertcat(X_R',ones(1,r));
minimize(norm((W - G*X_z),'fro') + 0.1*l1_mat(G))
cvx_end
G = G(:,1:dr);
G = (abs(G)>0.00001).*G;
[cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,G);
[accuracysign,cmsign] = CalcSign(G,Gtrue);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;result{method+1,14} = cm;result{method+1,15} = [cmsign];result{method+1,16} = [accuracysign];result{method+1,17} = [F1];
Winit = W;
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;
%% MTL -least Lasso
tic;
method = method+1;
obj_func_str = 'Least_Lasso';
eval_func_str = 'eval_rmse';
param_range = [0,0.001,0.01,0.1,1];
[best_param, perform_mat] = TuneParam21...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn,Ytrn,vadrate);
% build model using the optimal parameter
[W,Fval] = Least_Lasso(Xtrn, Ytrn, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
cvx_begin quiet
variable G(dl, dr+1)
X_z = vertcat(X_R',ones(1,r));
minimize(norm((W - G*X_z),'fro') + 0.1*l1_mat(G))
cvx_end
G = G(:,1:dr);
G = (abs(G)>0.00001).*G;
[cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,G);
[accuracysign,cmsign] = CalcSign(G,Gtrue);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;result{method+1,14} = cm;result{method+1,15} = [cmsign];result{method+1,16} = [accuracysign];result{method+1,17} = [F1];
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;
%% MTMLa_lasso
tic;
method = method+1;
obj_func_str = 'MTMLa';
eval_func_str = 'eval_rmse_MTMLa';
param_range = [0,0.001,0.1,1,5,10];
[best_param, perform_mat] = TuneParam_MTMLa...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn,Ytrn,vadrate,X_R); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% build model using the optimal parameter
[G,Fval] = MTMLa(Xtrn,Ytrn,X_R, best_param, opts);
G = (abs(G)>0.00001).*G;
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2_MTMLa(Xtst,Ytst,G,X_R);
for i = 1: length(Xtrn)
    W(:,i) = G'*X_R(i,:)';
end
[cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,G');
[accuracysign,cmsign] = CalcSign(G',Gtrue);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;result{method+1,14} = cm;result{method+1,15} = [cmsign];result{method+1,16} = [accuracysign];result{method+1,17} = [F1];

%% MTMLb
tic;
method = method+1;
obj_func_str = 'Least_Lasso';
eval_func_str = 'eval_rmse';
param_range = [0,0.001,0.01,0.1,1];
Xreg_trn = cell(1,r);
for i = 1:r
    tmp = Xtr_LR{i};
    tmp(:,dl+1:end) = tmp(:,dl+1:end)+randn(size(tmp,1),dr);
    Xreg_trn{i} = tmp(:,dl+1:end);
end
Xcsitrn = createX_csi_local(Xtrn,Xreg_trn);
Xtrn_noise = cell(1,r);

for i = 1:r
    Xtrn_noise{i} = horzcat(Xtrn{i},Xreg_trn{i},Xcsitrn{i});
end

[best_param, perform_mat] = TuneParam21...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn_noise,Ytrn,vadrate);
% build model using the optimal parameter
[W,Fval] = Least_Lasso(Xtrn_noise, Ytrn, best_param, opts);

Xreg_tst = cell(1,r);
for i = 1:r
    tmp = Xtst_LR{i};
    tmp(:,dl+1:end) = tmp(:,dl+1:end)+randn(size(tmp,1),dr)./10;
    Xreg_tst{i} = tmp(:,dl+1:end);
end
Xcsitst = createX_csi_local(Xtst,Xreg_tst);
Xtst_noise = cell(1,r);

for i = 1:r
    Xtst_noise{i} = horzcat(Xtst{i},Xreg_tst{i},Xcsitst{i});
end

[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst_noise,Ytst,W);
WrMTL = W(dl+dr+1:end,:);
G = median(WrMTL,2);
G = reshape(G,[dr,dl])'
[cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,G);
[accuracysign,cmsign] = CalcSign(G,Gtrue);
result{method+1,1} = 'MTML_randNoise'; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;result{method+1,15} = 'error_CSI';
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;
% 
% %% MTMLc
% tic;
% method = method+1;
% obj_func_str = 'Least_Lasso';
% eval_func_str = 'eval_rmse';
% param_range = [0,0.001,0.01,0.1,1];
% Xreg_trn = cell(1,r);
% for i = 1:r
%     tmp = Xtr_LR{i};
%     tmp(:,dl+1:end) = tmp(:,dl+1:end);
%     Xreg_trn{i} = tmp(:,dl+1:end);
% end
% Xcsitrn = createX_csi_local(Xtrn,Xreg_trn);
% Xtrn_noise = cell(1,r);
% 
% for i = 1:r
%     Xtrn_noise{i} = horzcat(Xtrn{i},Xreg_trn{i},Xcsitrn{i});
% end
% 
% [best_param, perform_mat] = TuneParam21...
%     (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn_noise,Ytrn,vadrate);
% % build model using the optimal parameter
% [W,Fval] = Least_Lasso(Xtrn_noise, Ytrn, best_param, opts);
% 
% Xreg_tst = cell(1,r);
% for i = 1:r
%     tmp = Xtst_LR{i};
%     tmp(:,dl+1:end) = tmp(:,dl+1:end);
%     Xreg_tst{i} = tmp(:,dl+1:end);
% end
% Xcsitst = createX_csi_local(Xtst,Xreg_tst);
% Xtst_noise = cell(1,r);
% 
% for i = 1:r
%     Xtst_noise{i} = horzcat(Xtst{i},Xreg_tst{i},Xcsitst{i});
% end
% 
% [rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst_noise,Ytst,W);
% WrMTL = W(dl+dr+1:end,:);
% G = median(WrMTL,2);
% G = reshape(G,[dr,dl])'
% [cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,G);
% [accuracysign,cmsign] = CalcSign(G,Gtrue);
% result{method+1,1} = 'MTML_NoNoise'; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
% result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;result{method+1,14} = cm;result{method+1,15} = cmsign;result{method+1,16} = [accuracysign];result{method+1,17} = [F1];
% clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;

%% cv_SolverL1
tic;
method = method+1;
obj_func_str = 'Least_Lasso';
eval_func_str = 'eval_rmse';
lambda1 = [0,0.01,0.1];
lambda2 = [0,0.01,0.1];
% [best_param, perform_mat] = TuneParam21...
%     (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn_noise,Ytrn,vadrate);
param_range =  generateParamset2(lambda1,lambda2);
Xreg_trn = cell(1,r);
for i = 1:r
    tmp = Xtr_LR{i};
    tmp(:,dl+1:end) = tmp(:,dl+1:end);
    Xreg_trn{i} = tmp(:,dl+1:end);
end
[Wl,gamma] =  cv_solver(Ytrn,Xtrn,Xreg_trn,1,0.01,randn(dl,dr),200,1e-9);
gammamat = repmat(gamma,[1,r]);
W = vertcat(Wl,gammamat);

Xreg_tst = cell(1,r);
for i = 1:r
    tmp = Xtst_LR{i};
    tmp(:,dl+1:end) = tmp(:,dl+1:end);
    Xreg_tst{i} = tmp(:,dl+1:end);
end
Xcsitst = createX_csi_local(Xtst,Xreg_tst);
Xtst_noise = cell(1,r);

for i = 1:r
    Xtst_noise{i} = horzcat(Xtst{i},Xreg_tst{i},Xcsitst{i});
end
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst_noise,Ytst,W);
G = reshape(gamma,[dr,dl])';
[cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,G);
[accuracysign,cmsign] = CalcSign(G,Gtrue);
result{method+1,1} = 'MTML_sameG'; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = 'best_param';result{method+1,11} = 'perform_mat';result{method+1,12} = 'Fval';result{method+1,13} = G;result{method+1,14} = cm;result{method+1,15} = cmsign;result{method+1,16} = [accuracysign];result{method+1,17} = [F1];
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;
fname = ['Syn9region3_','28_',num2str(ROUND)];
% save(fname);
result

% bar(cat(1,X_R_local{:}))
% figure
% bar(cat(1,Xlr{:}))
end 
norm(cat(1,Xlr{:})-cat(1,X_R_local{:}))
bar(cat(1,X_R_local{:}))
figure
bar(cat(1,Xlr{:}))
Gcli = Gcli./10;
G_estimateint = G_estimateint./10;
G_dirty = G_dirty./10;
G_Leastlasso = G_Leastlasso./10;





