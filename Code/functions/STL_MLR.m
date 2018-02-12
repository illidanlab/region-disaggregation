function W = STL_MLR(Xtrn, Ytrn, param,opts)
no_trn_idx = [];

for i = 1: length(Xtrn);
    if size(Ytrn{i},1) >=1
        W(:,i) = regress(Ytrn{i},Xtrn{i});
        %         W(:,i) = lasso(Xtrn{i},Ytrn{i},'lambda',param);
    else
        no_trn_idx = [no_trn_idx,i];
    end
end
tmp = setdiff(1:length(Xtrn),no_trn_idx);
W(:,no_trn_idx) = repmat(mean(W(:,tmp),2),1,length(no_trn_idx));
