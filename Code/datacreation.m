addpath(genpath('./MALSAR-master/MALSAR/functions/')); % load function
addpath('./MALSAR-master/MALSAR/utils/'); % load utilities
addpath('./functions/'); % load utilities
% addpath(genpath('/home/yuanshu2/MALSAR1.1/MALSAR/functions/')); % load function
% addpath('/home/yuanshu2/MALSAR1.1/MALSAR/utils/'); % load utilities
% addpath('functions');
% pctRunOnAll warning off
% mex segL2Proj.c
% mex segL2.c
for ROUND =1 :10 
%%  ========================================================================
%%  Preprocessing                                                          
%%  ========================================================================
tic;
% Load data
clearvars -except ROUND;
% load 'D:\sy\4MultiLevel\data\data1.mat';
load data3.mat;

% Keep related variables ------------------------------------
dataL = dataL(dataL(:,17)>=1 & ~isnan(dataL(:,17)),:); % keep lakes that has max depth >=1, keep where tp is not nan
% dataL = dataL(dataL(:,17)>=1 & ~isnan(dataL(:,17)) &~isnan(dataL(:,19)) & dataL(:,19)~=0,:); % keep lakes that has max depth >=1, keep where tp is not nan

dataL(:,end+1) = dataL(:,18)./dataL(:,16);

% var = 19; varname = 'tp'; 
% var = 20; varname = 'tn';
% var = 21; varname = 'chla';
var = 22; varname = 'secchi';
% var = 23; varname = 'no2no3'; 
tmp = dataL(:,var);
if(sum(tmp==0)~=0)
    fprintf('remove %i row(s) with %s value = 0\n',sum(tmp==0),varname);
    dataL = dataL(tmp~=0,:);
end

dataL = dataL(~isnan(dataL(:,var)),:); % remove rows that response var is NaN
y = log10(dataL(:,var));  % log10 of response variable 
y = (y-mean(y))./std(y); % standardization of response variable

dataL = [dataL(:,1),dataL(:,13:15),y,dataL(:,[2:12,17]),dataL(:,end)]; % 1-lakeid, 2-eduid, 3-lat, 4-lon,5-response, 6-end-predictor
% dataL = [dataL(:,1),dataL(:,13:15),y,dataL(:,[2:12,17]),log10(dataL(:,19)),dataL(:,end)]; % 1-lakeid, 2-eduid, 3-lat, 4-lon,5-response, 6-end-predictor

lakevar = lakedata.Properties.VariableNames';
lakevar = strrep(lakevar,'4ha_buffer500m_nlcd2001','');
lakevar = [lakevar(1);lakevar(13:15);lakevar(var);lakevar([2:12,17]);'iws/lakearea'];

% Standadize the Local predictor ------------------------------------
tmp = dataL(:,6:end);
% tmpidx = dataL(:,2);
% tmpeduidx = unique(tmpidx);
% for i = 1: length(tmpeduidx)
%     selecttmp = tmp(tmpidx == tmpeduidx(i),:);
%     if( size(selecttmp,1)>=2) % if only one data points then continue 
%         m = mean(selecttmp,1);
%         s = nanstd(selecttmp,[],1);
%         s(s==0) = eps;
%         tmp(tmpidx == tmpeduidx(i),:) = tmp(tmpidx == tmpeduidx(i),:) - repmat(m,size(selecttmp,1),1);
%         tmp(tmpidx == tmpeduidx(i),:) = (tmp(tmpidx == tmpeduidx(i),:) - repmat(m,size(selecttmp,1),1))...
%             ./repmat(s,size(selecttmp,1),1);
%     end
% end
tmp = (tmp- repmat(mean(tmp),size(tmp,1),1))./repmat(nanstd(tmp),size(tmp,1),1);
% tmp = (tmp- repmat(mean(tmp),size(tmp,1),1));
dataL = [dataL(:,1:5),tmp];
clear tmp tmpidx tmpeduidx selecttmp m i;

% Combine region data ------------------------------------
Eduid = unique(dataL(:,2));
dataR = dataR(ismember(dataR(:,1),Eduid),:);
dataR = sortrows(dataR,1);
% AGGREGATE HERE
dataRR = [dataR(:,1:7),sum(dataR(:,14:17),2),sum(dataR(:,18:20),2),sum(dataR(:,21:22),2),sum(dataR(:,23:24),2)];
dataR = dataRR;
tmp = dataR(:,4:end);
m = mean(tmp);
s = nanstd(tmp);
tmp = (tmp - repmat(m,size(tmp,1),1))./repmat(s,size(tmp,1),1);
% tmp = tmp - repmat(m,size(tmp,1),1);
dataR = [dataR(:,1:3),tmp]; % eduid, lat, lon, 4-end predictor;
regionvar = regiondata.Properties.VariableNames';
regionvar = strrep(regionvar,'surficialgeology','sg');
regionvar = strrep(regionvar,'nlcd2001_','');
regionvar = [regionvar(1:7);'developed';'forest';'algriculture';'wetland'];
clear tmp m;
X_R = dataR(:,4:end);
r = size(Eduid,1);
tp_X = cell(1,r);
tp_Y = cell(1,r);
tp_coord = cell(1,r);
for i = 1:r
    idx = find(dataL(:,2)==Eduid(i));
    tp_X{i} = dataL(idx,6:end);
    tp_Y{i} = y(idx);
    tp_coord{i} = dataL(idx,3:4);
end
tp_tmp = tp_X
save('secchi_X','tp_X')
save('secchi_Y','tp_Y')
save('secchi_coord','tp_coord')
% generate trn tst index ------------------------------------
trnrate = 2/8;vadrate = 1/2;
[Trnidx,Tstidx,TrnNum,TstNum] = GenerateTrnTstIdx(dataL(:,[1,2,5]),trnrate,ROUND);%1-lakeid,2-eduid,3-reponse
LatLonL = dataL(:,1:4); % 1-lakeid,2-eduid,3-lat,4-lon;
% add column of ones 
dataL = [dataL(:,[1,2]),dataL(:,5),ones(size(dataL,1),1),dataL(:,6:end)]; %1-lakeid,2-eduid,3-reponse, 4-end,predictor
lakevar = [lakevar([1,2,5]);'AllOne';lakevar(6:end)];
coord = LatLonL(:,3:4);
[Xtrn, Ytrn, Xtst, Ytst] = SplitTrnTst4(dataL,Trnidx ,Tstidx,coord);% data:1-lakeid, 2-eduid, 3-reponse, 4-end predictor
LatLonR = dataR(:,1:3);% 1-eduid,2-lat,3-lon;
dataR = dataR(:,4:end);% predictor
% add column of onesx 
dataR = [ones(size(dataR,1),1),dataR];
% regionvar = regionvar(4:end);
regionvar = ['AllOne';regionvar(4:end)];
clear lakedata regiondata;

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

data_stl = zeros(size(dataL,1),size(dataL,2)+size(dataR(:,2:end),2));
for i = 1: size(Eduid,1)
    data_stl(dataL(:,2) == Eduid(i),:) = [dataL(dataL(:,2)==Eduid(i),:),repmat(dataR(i,2:end),sum(dataL(:,2)==Eduid(i)),1)];
%     data_stl = [data_stl;[dataL(dataL(:,2)==Eduid(i),:),repmat(dataR(i,2:end),sum(dataL(:,2)==Eduid(i)),1)]];
end

% [Xtrn0, Ytrn0, Xtst0, Ytst0] = SplitTrnTst4(data_stl,Trnidx ,Tstidx);
%% One global model (lasso)
tic;
method = 1;
obj_func_str = 'STL-global model';
eval_func_str = 'eval_rmse';
% param_range = 0.01*(1:10);
[best_param, perform_mat] = TuneParam_lasso2(param_range,data_stl,Trnidx,vadrate);
w = lasso(data_stl(Trnidx,4:end),data_stl(Trnidx,3),'lambda',best_param);
ypred = data_stl(Tstidx,4:end)*w;
y = data_stl(Tstidx,3);
[r2,rmse] = rsquare(y,ypred);

rmseall= zeros(size(Eduid)); r2all = rmseall;
id_tmp = dataL(Tstidx,2);
for t = 1: size(Eduid,1);
    y_pred_t = ypred(id_tmp == Eduid(t));
    y_t = y(id_tmp ==Eduid(t));
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
end

result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = y;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;
clear y ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w id_tmp t;

%% One global model (lasso) without Region info
tic;
method = method+1;
obj_func_str = 'STL-global noregion';
eval_func_str = 'eval_rmse';
% param_range = 0.01*(1:10);
[best_param, perform_mat] = TuneParam_lasso2(param_range,dataL,Trnidx,vadrate);
w = lasso(dataL(Trnidx,4:end),dataL(Trnidx,3),'lambda',best_param);
ypred = dataL(Tstidx,4:end)*w;
y = dataL(Tstidx,3);
[r2,rmse] = rsquare(y,ypred);

rmseall= zeros(size(Eduid)); r2all = rmseall;
id_tmp = dataL(Tstidx,2);
for t = 1: size(Eduid,1);
    y_pred_t = ypred(id_tmp == Eduid(t));
    y_t = y(id_tmp ==Eduid(t));
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
end

result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = y;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;
clear y ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w id_tmp t;
%% STL - MLR (N - Independent model)
tic;
method = method+1;
obj_func_str = 'STL_MLR';
eval_func_str = 'eval_rmse';
% param_range = 10*(1:10);
[best_param, perform_mat] = TuneParam2...
    (obj_func_str, opts, 1, eval_func_str, higher_better,dataL,Trnidx,vadrate);
% build model using the optimal parameter
W = STL_MLR(Xtrn, Ytrn, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W ;
%% STL - Lasso (N - Independent model)
tic;
method = method+1;
obj_func_str = 'STL_lasso';
eval_func_str = 'eval_rmse';
% param_range = 10*(1:10);
[best_param, perform_mat] = TuneParam2...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,dataL,Trnidx,vadrate);
% build model using the optimal parameter
W = STL_lasso(Xtrn, Ytrn, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W ;
%% MTL - L21
tic;
method = method+1;
obj_func_str = 'Least_L21';
eval_func_str = 'eval_rmse';
param_range = [1:10,20:20:100];
[best_param, perform_mat] = TuneParam2...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,dataL,Trnidx,vadrate);
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
param_range = [1:10,20:20:100];
[best_param, perform_mat] = TuneParam2...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,dataL,Trnidx,vadrate);
% build model using the optimal parameter
[W,Fval] = Least_Lasso(Xtrn, Ytrn, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;
% Two level methods
%% MTMLa_lasso
tic;
method = method+1;
obj_func_str = 'MTMLa';
eval_func_str = 'eval_rmse_MTMLa';
param_range = [1:10,20:20:100];
[best_param, perform_mat] = TuneParam_MTMLa...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% build model using the optimal parameter
[G,Fval] = MTMLa(Xtrn, Ytrn,dataR, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2_MTMLa(Xtst,Ytst,G,dataR);
for i = 1: length(Xtrn)
    W(:,i) = G'*dataR(i,:)';
end
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;

%% MTMLb
param_range = [0.0001 0.001 0.01 0.1 1 10];
tic;
method = method+1;
obj_func_str = 'MTMLb2';
eval_func_str = 'eval_rmse';
rho1 = param_range;
rho2 = param_range;
rho3 = [1,5,10,20,50];
param_set = generateParamset(rho1,rho2,rho3);
[best_param, perform_mat] = TuneParam_MTMLb...
    (obj_func_str, opts, param_set, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% build model using the optimal parameter
[W, G,Fval] = MTMLb2(Xtrn, Ytrn, dataR,best_param(1),best_param(2),best_param(3),opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;result{method+1,7} = ypred;
result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;

%% MTMLb L21
param_range = [0.0001 0.001 0.01 0.1 1 10];
tic;
method = method+1;
obj_func_str = 'MTMLb3';
eval_func_str = 'eval_rmse';
rho1 = [1,5,10,20,50];
rho2 = param_range;
rho3 = param_range;
param_set = generateParamset(rho1,rho2,rho3);
[best_param, perform_mat] = TuneParam_MTMLb...
    (obj_func_str, opts, param_set, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% build model using the optimal parameter
[W, G,Fval] = MTMLb3(Xtrn, Ytrn, dataR,best_param(1),best_param(2),best_param(3),opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;result{method+1,7} = ypred;result{method+1,8} = yreal;
result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;
%% MTMLc
tic;
method = method+1;
obj_func_str = 'MTMLc2';
eval_func_str = 'eval_rmse';
opts.initW = Winit;
param_range = [0.01 1 100];
rho1 = 4:2:12;
rho2 = param_range;
rho3 = param_range;
rho4 = param_range;
rho5 = param_range;
param_set = combvec(rho1,rho2,rho3,rho4,rho5)';
[best_param, perform_mat] = TuneParam_MTMLc...
    (obj_func_str, opts, param_set, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
[U,V,R,Fval,W] = MTMLc2(Xtrn, Ytrn, dataR,best_param(1),best_param(2),best_param(3),...
    best_param(4),best_param(5),opts);
[rmse1,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
toc
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse1;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;
result{method+1,14} = U;result{method+1,15} = V;result{method+1,16} = R;result{method+1,13} = (U*R'*inv(R*R'))';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%search careful
% method = method + 1;
% rho1 = [best_param(1)-1  best_param(1) min(best_param(1)+1,14)];
% rho2 = [best_param(2)*0.1  best_param(2) best_param(2)*10];
% rho3 = [best_param(3)*0.1  best_param(3) best_param(3)*10];
% rho4 = [best_param(4)*0.1  best_param(4) best_param(4)*10];
% rho5 = [best_param(5)*0.1  best_param(5) best_param(5)*10];
% param_set = combvec(rho1,rho2,rho3,rho4,rho5)';
% tic
% [best_param, perform_mat] = TuneParam_MTMLc...
%     (obj_func_str, opts, param_set, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% [U,V,R,Fval,W] = MTMLc2(Xtrn, Ytrn, dataR,best_param(1),best_param(2),best_param(3),...
%     best_param(4),best_param(5),opts);
% [rmse2,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
% toc
% result{method+1,1} = obj_func_str; result{method+1,2} = toc;result{method+1,3} = rmse2;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
% result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;
% result{method+1,14} = U;result{method+1,15} = V;result{method+1,16} = R;result{method+1,13} = (U*R'*inv(R*R'))';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%search careful

% method = method+1;
% rho1 = [best_param(1)-1  best_param(1) min(best_param(1)+1,14)];
% rho2 = [best_param(2)*0.8  best_param(2) best_param(2)*1.2];
% rho3 = [best_param(3)*0.8  best_param(3) best_param(3)*1.2];
% rho4 = [best_param(4)*0.8  best_param(4) best_param(4)*1.2];
% rho5 = [best_param(5)*0.8  best_param(5) best_param(5)*1.2];
% param_set = combvec(rho1,rho2,rho3,rho4,rho5)';
% tic;
% [best_param, perform_mat] = TuneParam_MTMLc...
%     (obj_func_str, opts, param_set, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% [U,V,R,Fval,W] = MTMLc2(Xtrn, Ytrn, dataR,best_param(1),best_param(2),best_param(3),...
%     best_param(4),best_param(5),opts);
% [rmse3,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
% toc;
% result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse3;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
% result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;
% result{method+1,14} = U;result{method+1,15} = V;result{method+1,16} = R;result{method+1,13} = (U*R'*inv(R*R'))';
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval G U V R i rho1 rho2 rho3 rho4 rho5 myparam;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% toc;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Calibration
% param_range = [0.0001 0.001 0.01 0.1 1 10];
% tic;
% %method = method+1;
% obj_func_str = 'MLMT_Calibration';
% eval_func_str = 'eval_rmse';
% lambda1 = param_range;
% lambda2 = param_range;
% lambda3 = param_range;
% param_set = generateParamset(lambda1,lambda2,lambda3);
% X_L = data_stl(:,5:17);
% X_R = data_stl(:,18:end);
% y = data_stl(:,3);
% X_L_trainVali = X_L(Trnidx,:);
% X_R_trainVali = X_R(Trnidx,:);
% X_L_test = X_L(Tstidx,:);
% X_R_test = X_R(Tstidx,:);
% y_trainVali = y(Trnidx,:);
% y_testVali = y(Tstidx,:);
% X = horzcat(X_L_train,X_R_train);
% [trainInd,valInd] = dividerand(size(X_L_trainVali,1),0.5,0.5);
% X_L_Vali = X_L_trainVali(valInd,:);
% X_R_Vali = X_R_trainVali(valInd,:);
% y_Vali = y_trainVali(valInd,:);
% X_L_train = X_L_trainVali(trainInd,:);
% X_R_train = X_R_trainVali(trainInd,:);
% y_train = y_trainVali(trainInd,:);
% 
% [X_L_trainVali_cell,X_R_trainVali_cell,y_trainVali_cell] = frame2cell(X_L_train,X_R_train,y_train);
% [X_L_train_cell,X_R_train_cell,y_train_cell] = frame2cell(X_L_train,X_R_train,y_train);
% [X_L_vali_cell,X_R_vali_cell,y_vali_cell] = frame2cell(X_L_Vali,X_R_Vali,y_Vali);
% [X_L_test_cell,X_R_test_cell,y_test_cell] = frame2cell(X_L_test,X_R_test,y_test);
% 
% X_train_cell = cell(length(X_L_train_cell),1);
% for i = 1:length(X_L_train_cell)
%     X_train_cell{i} = horzcat(X_L_train_cell{i},X_R_train_cell{i});
% end
% 
% X_vali_cell = cell(length(X_L_vali_cell),1);
% for i = 1:length(X_L_vali_cell)
%     X_vali_cell{i} = horzcat(X_L_train_cell{i},X_R_train_cell{i});
% end
% 
% X_csi_train_cell = {};
% d_L = 13;
% d_R = 8;
% for i = 1:length(X_L_train_cell)
%        X_csi_train_cell{i} = zeros(size(X_L_train_cell{i},1),d_L*d_R);
%        t = 1;
%        for s = 1:d_L
%            for t = 1:d_R
%                
%                tmp = X_L_train_cell{i}(:,s).*X_R_train_cell{i}(:,t)
%                X_csi_cell{i}(:,t)=tmp;
%         
%            end         
%        end   
% end
% 
% X_csi_vali_cell = {};
% d_L = 13;
% d_R = 8;
% for i = 1:length(X_L_train_cell)
%        X_csi_vali_cell{i} = zeros(size(X_L_train_cell{i},1),d_L*d_R);
%        t = 1;
%        for s = 1:d_L
%            for t = 1:d_R
%                
%                tmp = X_L_train_cell{i}(:,s).*X_R_train_cell{i}(:,t)
%                X_csi_cell{i}(:,t)=tmp;
%         
%            end         
%        end   
% end
% 
% 
% [best_param, perform_mat] = TuneParam_MTML_Calibration...
%     (X_train_cell,y_train_cell,X_csi_train_cell, param_set,...
%     X_vali_cell,y_vali_cell,X_csi_vali_cell,X_test_cell,y_test_cell,X_csi_test_cell); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% % build model using the optimal parameter
% [W, G,Fval] = MTMLb3(Xtrn, Ytrn, dataR,best_param(1),best_param(2),best_param(3),opts);
% [rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
% result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;result{method+1,7} = ypred;result{method+1,8} = yreal;
% result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%param_range = [0.0001 0.001 0.01 0.1 1 10];

method = method+1;
obj_func_str = 'MTML_Calibration';
eval_func_str = 'eval_rmse';
lambda1_range = [0.0001,0.001,0.01,0.1,0.5,1];
lambda2_range = [0.0001,0.001,0.01,0.1,0.5,1];
lambda3_range = [0.0001,0.001,0.01,0.1,0.5,1,50];
param_set = generateParamset(lambda1_range,lambda2_range,lambda3_range);

tic;
[best_param, perform_mat] = TuneParam_MTML_Calibration...
    (obj_func_str, opts, param_set, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
toc;
% build model using the optimal parameter
Xcsitr = createX_csi(Xtrn,dataR);
Xtr_LR = cell(length(Xtrn),1);
for i = 1:length(Xtrn)
   Xtr_LR{i} = horzcat(Xtrn{i},repmat(dataR(i,:),size(Xtrn{i},1),1));
end
[ P, info, Th, q ] = BCD_DFISTA(Xtr_LR, Ytrn, best_param(1), best_param(2), best_param(3), Xcsitr);
Xcsitst = createX_csi(Xtst,dataR);
Xtst_LR = cell(length(Xtst),1);
for i = 1:length(Xtst)
   Xtst_LR{i} = horzcat(Xtst{i},repmat(dataR(i,:),size(Xtst{i},1),1));
end
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2_calibration(Xtst_LR,Ytst,Xcsitst,P,q);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;result{method+1,7} = ypred;result{method+1,8} = yreal;
result{method+1,9} = P;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = info.fvP(end);result{method+1,13} = q;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Not fix q
method = method+1;
obj_func_str = 'MTML_Calibration_notfixq';
eval_func_str = 'eval_rmse';
lambda1_range = [0.0001,0.001,0.01,0.1,0.5,1];
lambda2_range = [0.0001,0.001,0.01,0.1,0.5,1];
lambda3_range = [0.0001,0.001,0.01,0.1,0.5,1];
param_set = generateParamset(lambda1_range,lambda2_range,lambda3_range);

tic;
[best_param, perform_mat] = TuneParam_MTML_Calibration...
    (obj_func_str, opts, param_set, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
toc;
% build model using the optimal parameter
Xcsitr = createX_csi(Xtrn,dataR);
Xtr_LR = cell(length(Xtrn),1);
for i = 1:length(Xtrn)
   Xtr_LR{i} = horzcat(Xtrn{i},repmat(dataR(i,:),size(Xtrn{i},1),1));
end
[ P, info, Th, q ] = BCD_DFISTA(Xtr_LR, Ytrn, best_param(1), best_param(2), best_param(3), Xcsitr);
Xcsitst = createX_csi(Xtst,dataR);
Xtst_LR = cell(length(Xtst),1);
for i = 1:length(Xtst)
   Xtst_LR{i} = horzcat(Xtst{i},repmat(dataR(i,:),size(Xtst{i},1),1));
end
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2_calibration(Xtst_LR,Ytst,Xcsitst,P,q);
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;result{method+1,7} = ypred;result{method+1,8} = yreal;
result{method+1,9} = P;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = info.fvP(end);result{method+1,13} = q;

fname = ['result17_',varname,'28_',num2str(ROUND)];
save(fname);
result
end 

