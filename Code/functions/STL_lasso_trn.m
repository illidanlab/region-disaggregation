function W= STL_lasso_trn(Xtrn, Ytrn, best_param, opts)
no_trn_idx = [];
% lambdaset = zeros(length(Xtrn),1)
for i = 1: length(Xtrn);
    if size(Ytrn{i},1) > 1
          W(:,i) = lasso(Xtrn{i},Ytrn{i},'lambda',best_param(i));
    else
        no_trn_idx = [no_trn_idx,i];
    end
end    
end