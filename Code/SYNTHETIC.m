addpath(genpath('./MALSAR-master/MALSAR/functions/')); % load function
addpath('./MALSAR-master/MALSAR/utils/'); % load utilities
addpath('./functions/');
%% Data Generation
rng(1);
dz = 8; % feature dimensiton of level 2 variable
dw = 13; % feature dimension of level 1 variable
dcsi = dz*dw;
r = 10; % task number
N = randi([500,500],1,r); % sample size for each region
Wtrue = randn(dw,r);%initialize W
% for i = 1:r
%     W{i} = rand(dw+1,r);
% end
%alphatrue = rand(dw,r);
alphatrue = normrnd(100,10,[dw,1]);
alphatrue = repmat(alphatrue,[1,r]);
alphatrue = alphatrue + normrnd(0,1,[dw,r]);
betatrue = zeros(dz,1);%rand->zeros
betatrue = repmat(betatrue,[1,r]);
gammatrue = rand(dw*dz,1);
%Gtrue = rand(dz,dw);%initialize G
X_L = cell(1,r);
Y = cell(1,r);
X_R = randn(r,dz);

for i = 1:r
    X_L{i} = randn(N(i),dw);
end

for i = 1:r
    X_R(i,:) = randn(dz,1) + randi([-10,10],dz,1);
end

X_csi_syn = createX_csi(X_L,X_R);
noise2 = normrnd(0,5,[dw,r]);


%Wtrue = Gtrue'*X_R' + noise2;
Wtrue = mean(Wtrue,2);
for i = 1:r
%     Y{i} = X_L{i}*Wtrue(:,i) + normrnd(0,abs(randi([-3,3],1,1)),[size(X_L{i}*Wtrue(:,i),1),1]);
    %Y{i} = X_L{i}*Wtrue + normrnd(0,10,[size(X_L{i}*Wtrue(:,i),1),1]);+
    %normrnd(0,1,[N(1),1])
    Y{i} = X_L{i}*alphatrue(:,i) + (X_R(i,:) * betatrue(:,i))' + X_csi_syn{i}*gammatrue+normrnd(0,1,[N(1),1]);
end
Gtrue = (reshape(gammatrue,[8,13]))';

%% Experiment begins
tic;

varname = 'synthetic_samegammazero1_minusalpha';
% generate trn tst index ------------------------------------
trnrate = 0.9;vadrate = 0.8;
% the number of tasks

crossvalind('kfold',size(Y{1},1),10);%10 fold
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

Gcli = zeros(13,8);
G_Leastlasso = zeros(13,8);
G_estimateint = zeros(13,9);
G_dirty = zeros(13,8);

%%  ========================================================================
%%  Cross Validation Split data                                                          
%%  ========================================================================
for ROUND =1 :10 
 
tic;
% Load data
clearvars -except Gcli G_Leastlasso G_estimateint G_dirty ROUND alphatrue betatrue gammatrue X_csi_syn X_L X_R Y cvind Xtrn Xtst Ytrn Ytst r d dz dw dcsi varname Wtrue Gtrue;
% load 'D:\sy\4MultiLevel\data\data1.mat';
for i = 1:r
    testIdx = find(cvind{i}==ROUND);
    trainIdx = find(cvind{i}~=ROUND);
    Xtrn{i} = X_L{i}(trainIdx,:);
    Xtst{i} = X_L{i}(testIdx,:);
    Ytrn{i} = Y{i}(trainIdx,:);
    Ytst{i} = Y{i}(testIdx,:);  
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
for i = 1:13
    for j = 14:21
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
vadrate = 0.5;



%%  ========================================================================
%%  Initialization                                                          
%%  ========================================================================
result = cell(0);result{1,1} = 'obj func'; result{1,2} = 'runtime'; result{1,3} = 'rmse';result{1,4} = 'r2';
result{1,5} = 'rmse per region';result{1,6} = 'r2 per region';result{1,7} = 'ypred';result{1,8} = 'yreal';
result{1,9} = 'model w';result{1,10} = 'best param';result{1,11} = 'perform_mat';result{1,12} = 'funcFval';result{1,13} = 'model G';
higher_better = false;  % rmse is lower the better.
% param_range = [0.0001:0.0001:0.001,0.002:0.001:0.01,0.2:0.1:1];
param_range = [0,0.000001,0.001,0.01];
% param_range = [.1,.001];

% optimization options
opts.init = 2;      % guess start point from data.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance.
opts.maxIter = 500; % maximum iteration number of optimization.
opts.verbose = 0;
opts.OutermaxIter = 100;



% [Xtrn0, Ytrn0, Xtst0, Ytst0] = SplitTrnTst4(data_stl,Trnidx ,Tstidx);
% One global model (lasso)
tic;
method = 1;
obj_func_str = 'STL-global model';
eval_func_str = 'eval_rmse';
% param_range = 0.01*(1:10);
[best_param, perform_mat] = TuneParam_lasso2(param_range,XtrnG, YtrnG,vadrate);
w = lasso(XtrnG,YtrnG,'lambda',best_param);
ypred = XtstG*w;
[r2,rmse] = rsquare(YtstG,ypred);

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
[best_param, perform_mat] = TuneParam_lasso2(param_range,XtrnG(:,1:13), YtrnG,vadrate);
w = lasso(XtrnG(:,1:13),YtrnG,'lambda',best_param);
ypred = XtstG(:,1:13)*w;
[r2,rmse] = rsquare(YtstG,ypred);

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
G = w(22:end);
G = reshape(G,[dw,dz]);
Gcli = Gcli + G;
error_CSI = norm(G-reshape(gammatrue,[13,8]));
rmseall= zeros(r,1); r2all = zeros(r,1);
for t = 1: r
    y_pred_t = ypred(regiontst == t);
    y_t = YtstG(regiontst == t);
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
end

result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = YtstG;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,13} = G;result{method+1,15} = error_CSI;
clear ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w t G;
%% STL - MLR (N - Independent model)
tic;
method = method + 1;
obj_func_str =  'STL_MLR';
eval_func_str = 'eval_rmse';
% param_range = 0.01*(1:10);
Xtrn_csi = horzcat(XtrnG,Xtrncsi);
Xtst_csi = horzcat(XtstG,Xtstcsi);
[best_param, perform_mat] = TuneParam_lasso2(param_range,Xtrn_csi, YtrnG,vadrate);
w = lasso(Xtrn_csi,YtrnG,'lambda',best_param);
ypred = Xtst_csi*w;
[r2,rmse] = rsquare(YtstG,ypred);

rmseall= zeros(r,1); r2all = zeros(r,1);
for t = 1: r
    y_pred_t = ypred(regiontst == t);
    y_t = YtstG(regiontst == t);
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
end

result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = YtstG;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;
clear ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w t;
%% STL - Lasso (N - Independent model)
tic;
method = method+1;
obj_func_str = 'STL_lasso';
eval_func_str = 'eval_rmse';
% param_range = 10*(1:10);
[best_param, perform_mat] = TuneParam2...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn,Ytrn,vadrate);
% build model using the optimal parameter
W = STL_lasso(Xtrn, Ytrn, best_param, opts);
cvx_begin quiet
cvx_precision high
variable G(13, 9)
variable alphaestimate(13,10)
X_z = vertcat(X_R',ones(1,10))
%minimize(norm((W-alphaestimate) - G*X_z,'fro') + norm(alphaestimate - alphatrue,'fro'))
minimize(norm((W - G*X_z),'fro'))
cvx_end
error_CSI = norm(G(:,1:8)-reshape(gammatrue,[8,13])');
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,13} = G;
result{method+1,15} = error_CSI;
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W G;
%% MTL - L21
tic;
method = method+1;
obj_func_str = 'Least_L21';
eval_func_str = 'eval_rmse';
param_range = [0,0.01,0.1,1:10,20:20:100];
[best_param, perform_mat] = TuneParam2...
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
[best_param, perform_mat] = TuneParam2...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,Xtrn,Ytrn,vadrate);
% build model using the optimal parameter
[W,Fval] = Least_Lasso(Xtrn, Ytrn, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
cvx_begin quiet
cvx_precision high
variable G(13, 8)
minimize(norm((W-alphatrue) - G*X_R','fro'))
error_CSI = norm(G-reshape(gammatrue,[8,13])');
cvx_end
G_Leastlasso = G_Leastlasso + G;
cvx_begin quiet
cvx_precision high
variable Ges(13, 9)
variable alphaestimate(13,10)
X_z = vertcat(X_R',ones(1,10))
%minimize(norm((W-alphaestimate) - G*X_z,'fro') + norm(alphaestimate - alphatrue,'fro'))
minimize(norm((W - Ges*X_z),'fro'))
cvx_end
G_estimateint = G_estimateint + Ges;
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;result{method+1,15} = error_CSI;
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;

%% Dirty Model
method = method+1;
obj_func_str = 'Dirty_model';
eval_func_str = 'eval_rmse';
lambda1_range = [0,0.01,1:10,20:20:100];
lambda2_range = [0,0.01,0.1,1,10,50,100];
param_set = generateParamset2(lambda1_range,lambda2_range);
[best_param, perform_mat] = TuneParamDirty...
    (obj_func_str, opts, param_set, eval_func_str, higher_better,Xtrn,Ytrn,vadrate);
% build model using the optimal parameter
% Gfin = zeros(13,8)
[W,Fval, P, Q] = Nested_Dirty(Xtrn, Ytrn, best_param(1),best_param(2), opts);
cvx_begin quiet
cvx_precision high
variable G(13, 8)
variable alphaestimate(13,10)
minimize(norm((P-alphaestimate) - G*X_R','fro') + norm(alphaestimate - alphatrue,'fro'))
cvx_end
G_dirty = G_dirty + G;
% Gfin = Gfin + G
error_CSI = norm(G-reshape(gammatrue,[8,13])');
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;result{method+1,14} = Q;result{method+1,15} = error_CSI;
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;


% Two level methods
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

% %% MLMT_Calibration
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %param_range = [0.0001 0.001 0.01 0.1 1 10];
% method = method+1;
% obj_func_str = 'MTML_Calibration';
% eval_func_str = 'eval_rmse';
% lambda1_range = [0.00001,0.01,1];
% lambda2_range = [0.1,10,100];
% lambda3_range = [1,500,1000];
% param_set = generateParamset(lambda1_range,lambda2_range,lambda3_range);
% 
% tic;
% [best_param, perform_mat] = TuneParam_MTML_Calibration...
%     (obj_func_str, opts, param_set, eval_func_str, higher_better,Xtrn,Ytrn,vadrate,X_R); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% toc;
% % build model using the optimal parameter
% Xcsitr = createX_csi(Xtrn,X_R);
% Xtr_LR = cell(length(Xtrn),1);
% for i = 1:length(Xtrn)
%    Xtr_LR{i} = horzcat(Xtrn{i},repmat(X_R(i,:),size(Xtrn{i},1),1));
% end
% [ P, info, Th, q ] = BCD_DFISTA(Xtr_LR, Ytrn, best_param(1), best_param(2), best_param(3), Xcsitr);
% Xcsitst = createX_csi(Xtst,X_R);
% Xtst_LR = cell(length(Xtst),1);
% for i = 1:length(Xtst)
%    Xtst_LR{i} = horzcat(Xtst{i},repmat(X_R(i,:),size(Xtst{i},1),1));
% end
% [rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2_calibration(Xtst_LR,Ytst,Xcsitst,P,q);
% result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;result{method+1,7} = ypred;result{method+1,8} = yreal;
% result{method+1,9} = P;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = info.fvP(end);result{method+1,13} = q;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% method = method+1;
% obj_func_str = 'MTML_Calibration';
% eval_func_str = 'eval_rmse';
% lambda1_range = [best_param(1)*0.1,best_param(1),best_param(1)*10];
% lambda2_range = [best_param(2)*0.1,best_param(2),best_param(2)*10];
% lambda3_range = [best_param(3)*0.1,best_param(3),best_param(3)*10];
% param_set = generateParamset(lambda1_range,lambda2_range,lambda3_range);
% 
% tic;
% [best_param, perform_mat] = TuneParam_MTML_Calibration...
%     (obj_func_str, opts, param_set, eval_func_str, higher_better,Xtrn,Ytrn,vadrate,X_R); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% toc;
% % build model using the optimal parameter
% Xcsitr = createX_csi(Xtrn,X_R);
% Xtr_LR = cell(length(Xtrn),1);
% for i = 1:length(Xtrn)
%    Xtr_LR{i} = horzcat(Xtrn{i},repmat(X_R(i,:),size(Xtrn{i},1),1));
% end
% [ P, info, Th, q ] = BCD_DFISTA(Xtr_LR, Ytrn, best_param(1), best_param(2), best_param(3), Xcsitr);
% Xcsitst = createX_csi(Xtst,X_R);
% Xtst_LR = cell(length(Xtst),1);
% for i = 1:length(Xtst)
%    Xtst_LR{i} = horzcat(Xtst{i},repmat(X_R(i,:),size(Xtst{i},1),1));
% end
% [rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2_calibration(Xtst_LR,Ytst,Xcsitst,P,q);
% result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;result{method+1,7} = ypred;result{method+1,8} = yreal;
% result{method+1,9} = P;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = info.fvP(end);result{method+1,13} = q;



%% MTMLc
% tic;
% method = method+1;
% obj_func_str = 'MTMLc2';
% eval_func_str = 'eval_rmse';
% opts.initW = Winit;
% param_range = [0.01 1 100];
% rho1 = 4:2:12;
% rho2 = param_range;
% rho3 = param_range;
% rho4 = param_range;
% rho5 = param_range;
% param_set = combvec(rho1,rho2,rho3,rho4,rho5)';
% [best_param, perform_mat] = TuneParam_MTMLc...
%     (obj_func_str, opts, param_set, eval_func_str, higher_better,Xtrn,Ytrn,vadrate,X_R); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% [U,V,R,Fval,W] = MTMLc2(Xtrn, Ytrn, X_R,best_param(1),best_param(2),best_param(3),...
%     best_param(4),best_param(5),opts);
% [rmse1,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
% toc
% result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse1;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
% result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;
% result{method+1,14} = U;result{method+1,15} = V;result{method+1,16} = R;result{method+1,13} = (U*R'*inv(R*R'))';
% 
% method = method + 1;
% rho1 = [best_param(1)-1  best_param(1) min(best_param(1)+1,14)];
% rho2 = [best_param(2)*0.1  best_param(2) best_param(2)*10];
% rho3 = [best_param(3)*0.1  best_param(3) best_param(3)*10];
% rho4 = [best_param(4)*0.1  best_param(4) best_param(4)*10];
% rho5 = [best_param(5)*0.1  best_param(5) best_param(5)*10];
% param_set = combvec(rho1,rho2,rho3,rho4,rho5)';
% tic
% [best_param, perform_mat] = TuneParam_MTMLc...
%     (obj_func_str, opts, param_set, eval_func_str, higher_better,Xtrn,Ytrn,vadrate,X_R); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% [U,V,R,Fval,W] = MTMLc2(Xtrn, Ytrn, X_R,best_param(1),best_param(2),best_param(3),...
%     best_param(4),best_param(5),opts);
% [rmse2,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
% toc
% result{method+1,1} = obj_func_str; result{method+1,2} = toc;result{method+1,3} = rmse2;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
% result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;
% result{method+1,14} = U;result{method+1,15} = V;result{method+1,16} = R;result{method+1,13} = (U*R'*inv(R*R'))';
% 
% % method = method+1;
% % rho1 = [best_param(1)-1  best_param(1) min(best_param(1)+1,14)];
% % rho2 = [best_param(2)*0.8  best_param(2) best_param(2)*1.2];
% % rho3 = [best_param(3)*0.8  best_param(3) best_param(3)*1.2];
% % rho4 = [best_param(4)*0.8  best_param(4) best_param(4)*1.2];
% % rho5 = [best_param(5)*0.8  best_param(5) best_param(5)*1.2];
% % param_set = combvec(rho1,rho2,rho3,rho4,rho5)';
% % tic;
% % [best_param, perform_mat] = TuneParam_MTMLc...
% %     (obj_func_str, opts, param_set, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% % [U,V,R,Fval,W] = MTMLc2(Xtrn, Ytrn, dataR,best_param(1),best_param(2),best_param(3),...
% %     best_param(4),best_param(5),opts);
% % [rmse3,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
% % toc;
% % result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse3;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
% % result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;
% % result{method+1,14} = U;result{method+1,15} = V;result{method+1,16} = R;result{method+1,13} = (U*R'*inv(R*R'))';
% clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval G U V R i rho1 rho2 rho3 rho4 rho5 myparam;
% 
fname = ['result17_',varname,'28_',num2str(ROUND)];
save(fname);
result
end 
Gcli = Gcli./10;
G_estimateint = G_estimateint./10;
G_dirty = G_dirty./10;
G_Leastlasso = G_Leastlasso./10;
