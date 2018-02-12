function W = STL_lasso2(Xtrn, Ytrn, param,opts)
% [W3,status]=l1_ls(a{1},b{1},best_param,1e-5,1);
no_trn_idx = [];
for i = 1: length(Xtrn);
    if size(Ytrn{i},1) > 1
        W(:,i) = l1_ls(Xtrn{i},Ytrn{i},param,1e-5,1);
    else
        no_trn_idx = [no_trn_idx,i];
    end
end
tmp = setdiff(1:length(Xtrn),no_trn_idx);
% W(:,no_trn_idx) = repmat(mean(W(:,tmp),2),1,length(no_trn_idx));
