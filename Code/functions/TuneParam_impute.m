function [ best_idx, perform_mat] = TuneParam_impute...
    (param_set, X, Y,X_R,coord,trainIdx,testIdx,vadratio)
%% INPUT
% data: original data
%   obj_func_str:  1-parameter optimization algorithms
%   param_set:   the set of the parameter. array
%   eval_func_str: evaluation function:
%       signature [performance_measure] = eval_func(Y_test, X_test, W_learnt)
%   higher_better: if the performance is better given
%           higher measurement (e.g., Accuracy, AUC)
%% OUTPUT
%   best_param:  best parameter in the given parameter range
%   perform_mat: the average performance for every parameter in the
%                parameter range.
% eval_func = str2func(eval_func_str);
% obj_func  = str2func(obj_func_str);

% performance vector
perform_mat = zeros(size(param_set,1),1);
r = length(Y);
dl = size(X{1},2);
dr = size(X_R,2);
% delete(gcp('nocreate'));
% obj = parpool(24);
for p_idx = 1: length(perform_mat)
    [~,~,Gestimate,Xlr,ff] = MTML_imputation_fs(Y,X,X_R,param_set(p_idx,1),...,
        param_set(p_idx,2),param_set(p_idx,3),param_set(p_idx,4),coord,randn(dl,dr),50,1e-5);
    [cv_Xtr,cv_Ytr,cv_Xte,cv_Yte,cv_Xlrtr,cv_Xlrte,trnindex,tstindex] = splitTrnVal_3(X,Xlr,Y,vadratio);
    [Wl,gamma] =  cv_solver(cv_Ytr,cv_Xtr,cv_Xlrtr,0,0,Gestimate,50,1e-5);
%     [Wl,gamma] =  cv_solver(Y,X,Xlr,0.001,param_set(p_idx,2),Gestimate,50,1e-3);
    xall = cell(1,r);
    GestimateBi = double(Gestimate~=0);
    gammabi = reshape(GestimateBi',[dl*dr,1]);
    indnzero = find(gammabi~=0);
    Xlrcsite = createX_csi_local(cv_Xte,cv_Xlrte);
    for i = 1:r
        Xlrcsite{i} = Xlrcsite{i}(:,indnzero);
    end
%   
%     ytmp = cv_Yte;
%     for i = 1:r
%         ytmp{i} = ytmp{i} - Xlrcsite{i}*gamma;
%     end
    for i = 1:r
        xall{i} = horzcat(cv_Xte{i},cv_Xlrte{i});
    end
    ypredcv = cell(1,r);
%     b = Least_Lasso(xall,cv_Yte,0);
%     b = zeros(size(xall{1},2),r)

    for i = 1:r
        ypredcv{i} = xall{i}*Wl(:,i) + Xlrcsite{i}*gamma;
    end
    Ytstcvall = cat(1,cv_Yte{:});
    Ypredtstcvall =cat(1,ypredcv{:});
    perform_mat(p_idx) = norm(Ytstcvall-Ypredtstcvall)/length(Ytstcvall);
end
% delete(obj);
% perform_mat = mean(perform_mat,2);
[~,best_idx] = min(perform_mat);
end

