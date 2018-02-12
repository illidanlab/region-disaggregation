function [lambdset] = TuneParam2...
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
obj_func  = str2func(obj_func_str);

[cv_Xtr, cv_Ytr, cv_Xte, cv_Yte] = splitTrnVal(X,Y,vadratio);
[W,lambdset] = obj_func(cv_Xtr, cv_Ytr, cv_Xte,cv_Yte);



end

% performance vector
% perform_mat = zeros(length(param_range),1);


