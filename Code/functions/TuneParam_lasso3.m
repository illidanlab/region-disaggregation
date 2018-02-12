function [ best_param, perform_mat] = TuneParam_lasso3...
    (param_range,data,Trnidx,vadratio)
%% INPUT
% data: original data
% YearCutoff: Year after for test.
% YearVad: Year after for validation.
%   obj_func_str:  1-parameter optimization algorithms
%   param_range:   the range of the parameter. array
%   eval_func_str: evaluation function:
%       signature [performance_measure] = eval_func(Y_test, X_test, W_learnt)
%   higher_better: if the performance is better given
%           higher measurement (e.g., Accuracy, AUC)
%% OUTPUT
%   best_param:  best parameter in the given parameter range
%   perform_mat: the average performance for every parameter in the
%                parameter range.
% Dec.9 
% Change the split criteria

% performance vector
perform_mat = zeros(length(param_range),1);
tmp1 = Trnidx(randperm(length(Trnidx)));
tmp2 = round(length(Trnidx)*vadratio);
Newvadidx = tmp1(1:tmp2);
Newtrnidx = setdiff(Trnidx,Newvadidx);
cv_Xtr = data(Newtrnidx,4:end);
cv_Ytr = data(Newtrnidx,3);
cv_Xte = data(Newvadidx ,4:end);
cv_Yte = data(Newvadidx,3);
for p_idx = 1: length(param_range)
%     w =lasso(cv_Xtr,cv_Ytr,'lambda',param_range(p_idx));
    w = l1_ls(cv_Xtr,cv_Ytr,param_range(p_idx),1e-5,1);
    ypred = cv_Xte*w;
    perform_mat(p_idx) = sqrt(1/length(ypred)*sum((ypred-cv_Yte).^2));
end

[~,best_idx] = min(perform_mat);
best_param = param_range(best_idx);
end

