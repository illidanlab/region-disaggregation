function [ best_param, perform_mat] = TuneParam21...
    ( obj_func_str, obj_func_opts, param_range, eval_func_str, higher_better,X,Y,vadratio)
%% INPUT
% data: original data
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
eval_func = str2func(eval_func_str);
obj_func  = str2func(obj_func_str);

% performance vector
perform_mat = zeros(length(param_range),1);

[cv_Xtr, cv_Ytr, cv_Xte, cv_Yte] = splitTrnVal(X,Y,vadratio);
for p_idx = 1: length(param_range)
    W = obj_func(cv_Xtr, cv_Ytr,param_range(p_idx), obj_func_opts);
    perform_mat(p_idx) = eval_func(cv_Xte,cv_Yte,W);
end

if(higher_better)
    [~,best_idx] = max(perform_mat);
else
    [~,best_idx] = min(perform_mat);
end
best_param = param_range(best_idx);
end