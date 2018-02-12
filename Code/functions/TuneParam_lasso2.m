function [ best_param, perform_mat] = TuneParam_lasso2...
    (param_range,dataX, dataY,vadratio)
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

N = size(dataX,1);
indperm = randperm(N);
X = dataX(indperm,:);
Y = dataY(indperm,:);

cv_Xtr = X(1:floor((1-vadratio)*N),:);
cv_Ytr = Y(1:floor((1-vadratio)*N),:);
cv_Xte = X(ceil((1-vadratio)*N):end,:);
cv_Yte = Y(ceil((1-vadratio)*N):end,:);
for p_idx = 1: length(param_range)
    w =lasso(cv_Xtr,cv_Ytr,'lambda',param_range(p_idx));
    ypred = cv_Xte*w;
    perform_mat(p_idx) = sqrt(1/length(ypred)*sum((ypred-cv_Yte).^2));
end

[~,best_idx] = min(perform_mat);
best_param = param_range(best_idx);
end

