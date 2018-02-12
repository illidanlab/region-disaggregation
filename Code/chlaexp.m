addpath(genpath('./MALSAR-master/MALSAR/functions/')); % load function
addpath('./MALSAR-master/MALSAR/utils/'); % load utilities
addpath('./functions/');
% addpath(genpath('/home/yuanshu2/MALSAR1.1/MALSAR/functions/')); % load function
% addpath('/home/yuanshu2/MALSAR1.1/MALSAR/utils/'); % load utilities
% addpath('functions');
% pctRunOnAll warning off
%mex prf_lbm.cpp
%mex segL2Proj.c
%mex segL2.c

%%  ========================================================================
%%  Readdata                                                         
%%  ========================================================================
tic;

% Keep related variables ------------------------------------
%varname = 'tp'; 
%varname = 'tn';
varname = 'chla';
%varname = 'secchi';
 
X_L = load(strcat('./Datareal/',varname,'_X.mat'));
X_R = load(strcat('./Datareal/',varname,'_X_R.mat'));
Y = load(strcat('./Datareal/',varname,'_Y.mat'));
coord = load(strcat('./Datareal/',varname,'_coord.mat'));

X_L = X_L.tp_X;
X_R = X_R.chla_X_R;
Y = Y.tp_Y;
coord = coord.tp_coord;
% X_L = eval(strcat('X_L.','tp','_X'));
% X_R = eval(strcat('X_R.','tp','_X_R'));
% Y = eval(strcat('Y.','tp','_Y'));
% coord = eval(strcat('coord.','tp','_coord'));
%the minimum sample size
threshold = 20;

ind = 1:size(Y,2);
for i = 1:size(Y,2)
    if size(Y{i},1) < threshold
        ind(i) = 0;   
    end
end
ind = find(ind>0);


X_L = X_L(:,ind);
X_R = X_R(ind,:);
Y = Y(:,ind);
coord = coord(:,ind);
% generate trn tst index ------------------------------------
trnrate = 0.9;vadrate = 1/2;
% the number of tasks
r = size(X_L,2);
dl = size(X_L{1},2);
dr = size(X_R,2);
crossvalind('kfold',size(Y{1},1),10);
Xtrn = cell(1,r);
Ytrn = cell(1,r);
Xtst = cell(1,r);
Ytst = cell(1,r);

cvind = cell(1,size(X_L,2));
for i = 1:size(X_L,2)
   cvind{i} =  crossvalind('kfold',size(Y{i},1),10);
end

d = size(Xtrn{1},2)-1;
dcsi = size(X_R,2);



%%  ========================================================================
%%  Cross Validation Split data                                                          
%%  ========================================================================
for ROUND =1 :10 
 
tic;
% Load data
clearvars -except ROUND X_L X_R Y dl dr coord cvind Xtrn Xtst Ytrn Ytst r d dcsi varname;
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
%get the region idx
Rtrnidx = zeros(size(XtrnG,1),1);
Rtstidx = zeros(size(XtstG,1),1);
samplesize_tst = zeros(r,1);

for i = 1:r
    samplesize_tst(i) = size(Xtst{i},1);
end

regiontst = createregion(samplesize_tst);


%validation rate
vadrate = 0.3;



%%  ========================================================================
%%  Initialization                                                          
%%  ========================================================================
result = cell(0);result{1,1} = 'obj func'; result{1,2} = 'runtime'; result{1,3} = 'rmse';result{1,4} = 'r2';
result{1,5} = 'rmse per region';result{1,6} = 'r2 per region';result{1,7} = 'ypred';result{1,8} = 'yreal';
result{1,9} = 'model w';result{1,10} = 'best param';result{1,11} = 'perform_mat';result{1,12} = 'funcFval';result{1,13} = 'model G';
higher_better = false;  % rmse is lower the better.
% param_range = [0.0001:0.0001:0.001,0.002:0.001:0.01,0.2:0.1:1];
param_range = [0.001,0.01];
% param_range = [.1,.001];

% optimization options
opts.init = 2;      % guess start point from data.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance.
opts.maxIter = 500; % maximum iteration number of optimization.
opts.verbose = 0;
opts.OutermaxIter = 100;



% [Xtrn0, Ytrn0, Xtst0, Ytst0] = SplitTrnTst4(data_stl,Trnidx ,Tstidx);
%% MTML_imputatin

tic;
method = 1;
lambda1_range = [10];
lambda2_range = [0.1];
lambda3_range = [0.1];
lambda4_range = [30];
param_set = generateParamset4(lambda1_range,lambda2_range,lambda3_range,lambda4_range);
[bestpara_ind,perform_mat] = TuneParam_impL21(param_set,Xtrn,Ytrn,X_R,coordtrn,trainIdx,testIdx,vadrate);
Ypredtst = cell(1,r);
bestpara = param_set(bestpara_ind,:);
[Wl,Wr,Gestimate,Xlr,Xlrtst,fval] = MTML_L21(Ytrn,Xtrn,Xtst,X_R,bestpara(1),bestpara(2),bestpara(3),bestpara(4),coord,randn(dl,dr),trainIdx,testIdx,50,1e-4);
%two stage
Xlrtrn = cell(1,r);
for i = 1:r
    Xlrtrn{i} = Xlr{i}(trainIdx{i},:);
end
xall = cell(1,r);
GestimateBi = double(Gestimate~=0);
gammabi = reshape(GestimateBi',[dl*dr,1]);
indnzero = find(gammabi~=0);
Xlrcsitrn = createX_csi_local(Xtrn,Xlrtrn);
% for i = 1:r
%     Xlrcsitrn{i} = Xlrcsitrn{i}(:,indnzero);
% end
for i = 1:r
    xall{i} = horzcat(Xtrn{i},Xlrtrn{i},Xlrcsitrn{i},ones(size(Xtrn{i},1),1));
end
b = Least_L21(xall,Ytrn,10);
Xlrcsite = createX_csi_local(Xtst,Xlrtst);
for i = 1:r
    xall{i} = horzcat(Xtst{i},Xlrtst{i},Xlrcsite{i},ones(size(Xtst{i},1),1));
end
for i = 1:r
    Ypredtst{i} = xall{i}*b(:,i);
end
% for i = 1:r
%     Ypredtst{i} = xall{i}*b(:,i);
% end
% for i = 1:r
%     Ypredtst{i} = Xtst{i}*Wl(:,i) +  Xlrtst{i} * Wr(:,i) + diag(Xtst{i}*Gestimate*Xlrtst{i}');
% end

rmseall= zeros(r,1); r2all = zeros(r,1);
for t = 1: r
    y_pred_t = Ypredtst{t};
    y_t = Ytst{t};
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);   
%     rmseall(t) = norm(y_t-y_pred_t);
end
Ypredtstall =cat(1,Ypredtst{:});
Ytstall = cat(1,Ytst{:});
%rmse = norm(Ytstall-Ypredtstall)/length(Ytstall);
[r2,rmse] = rsquare(Ytstall,Ypredtstall);

result{method+1,1} = 'MTML_imputation'; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = Ypredtst;result{method+1,8} = Ypredtstall;result{method+1,9} = Wl;result{method+1,10} = bestpara;result{method+1,11} = perform_mat;result{method+1,13} = Gestimate;
clear param_set;
%% One global model (lasso)
tic;
method = method + 1;
obj_func_str = 'STL-global model';
eval_func_str = 'eval_rmse';
% param_range = 0.01*(1:10);
[best_param, perform_mat] = TuneParam_lasso2(param_range,XtrnG, YtrnG,vadrate);
w = lasso(XtrnG,YtrnG,'lambda',best_param);
ypred = XtstG*w;
[r2,rmse] = rsquare(YtstG,ypred);
%rmse = norm(YtstG-ypred)/length(ypred);
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
G = reshape(G,[dl,dr]);
% [cm, accuracy, precision, recall, F1] = CalcSparse(Gtrue,G);
% [accuracysign,cmsign] = CalcSign(Gestimate,G);
% Gcli = Gcli + G;

% error_CSI = norm(G-reshape(gammatrue,[dl,dr]));
rmseall= zeros(r,1); r2all = zeros(r,1);
for t = 1: r
    y_pred_t = ypred(regiontst == t);
    y_t = YtstG(regiontst == t);
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
end

result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = YtstG;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,13} = G;
clear ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w t G;
%% STL - Lasso (N - Independent model)
tic;
method = method+1;
obj_func_str = 'STL_lasso';
eval_func_str = 'eval_rmse';
% param_range = 10*(1:10);
[lambdaset] = TuneParam2...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn,Ytrn,vadrate);
% build model using the optimal parameter
W = STL_lasso_trn(Xtrn, Ytrn, lambdaset, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = lambdaset;result{method+1,11} = 'perform_mat';result{method+1,13} = 'G';
result{method+1,15} = 'error_CSI';
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
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;
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
% cvx_begin quiet
% cvx_precision high
% variable G(dl, dr)
% minimize(norm((W-alphatrue) - G*X_R','fro'))
% error_CSI = norm(G-reshape(gammatrue,[dr,dl])');
% cvx_end
% G_Leastlasso = G_Leastlasso + G;
% cvx_begin quiet
% cvx_precision high
% variable Ges(dl, dr+1)
% variable alphaestimate(dl,r)
% X_z = vertcat(X_R',ones(1,r));
% %minimize(norm((W-alphaestimate) - G*X_z,'fro') + norm(alphaestimate - alphatrue,'fro'))
% minimize(norm((W - Ges*X_z),'fro'))
% cvx_end
% G_estimateint = G_estimateint + Ges(:,1:dr);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = 'G';result{method+1,15} = 'error_CSI';
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;
%% MTMLa_lasso
tic;
method = method+1;
obj_func_str = 'MTMLa';
eval_func_str = 'eval_rmse_MTMLa';
param_range = [0,0.001,0.1,1];
[best_param, perform_mat] = TuneParam_MTMLa...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn,Ytrn,vadrate,X_R); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% build model using the optimal parameter
[G,Fval] = MTMLa(Xtrn,Ytrn,X_R, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2_MTMLa(Xtst,Ytst,G,X_R);
for i = 1: length(Xtrn)
    W(:,i) = G'*X_R(i,:)';
end
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;

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

result{method+1,1} = 'MTML_randNoise'; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = 'G';result{method+1,15} = 'error_CSI';
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;

%% MTML_mean
tic;
method = method+1;
obj_func_str = 'MTML_sameG';
eval_func_str = 'eval_rmse_MTMLa';
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
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = 'best_param';result{method+1,11} = 'perform_mat';result{method+1,12} = 'Fval';result{method+1,13} = 'G';
fname = ['result17_',varname,'new',num2str(ROUND)];
save(fname);
result
end 

