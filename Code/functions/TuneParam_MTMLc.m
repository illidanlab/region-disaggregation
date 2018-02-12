function [ best_param, perform_mat] = TuneParam_MTMLc...
    ( obj_func_str, obj_func_opts, param_set, eval_func_str, higher_better,X,Y,vadratio,XR)
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
%pctRunOnAll warning off

eval_func = str2func(eval_func_str);
obj_func  = str2func(obj_func_str);

% performance vector
perform_mat = zeros(size(param_set,1),1);

[cv_Xtr, cv_Ytr, cv_Xte, cv_Yte] = splitTrnVal(X,Y,vadratio);
delete(gcp('nocreate'))
obj=parpool(24);
parfor p_idx = 1: size(param_set,1)
    [~,~,~,~,W] = obj_func(cv_Xtr, cv_Ytr, XR,param_set(p_idx,1),param_set(p_idx,2),param_set(p_idx,3),...
        param_set(p_idx,4),param_set(p_idx,5),obj_func_opts);
    perform_mat(p_idx) = eval_func(cv_Xte,cv_Yte,W);
end
delete(obj);
if(higher_better)
    [~,best_idx] = max(perform_mat);
else
    [~,best_idx] = min(perform_mat);
end
best_param = param_set(best_idx,:);

% for jj = 1: size(param_set,2)
%     param_range = unique(param_set(:,jj));
%     meanparam = zeros(1,length(param_range));
%     for ii = 1: length(param_range)
%         meanparam(1,ii) = mean(perform_mat(param_set(:,jj)==param_range(ii)));
%     end
%     if(higher_better)
%         [~,best_idx] = max(meanparam(:));
%     else
%         [~,best_idx] = min(meanparam(:));
%     end
%     best_param(jj) = param_range(best_idx);
% end

end

