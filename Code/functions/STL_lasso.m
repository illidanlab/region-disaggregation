function [W,lambdaset] = STL_lasso(Xtrn, Ytrn, Xte,Yte)
no_trn_idx = [];
lambdaset = zeros(length(Xtrn),1);
for i = 1: length(Xtrn);
    if size(Ytrn{i},1) > 1
%         W(:,i) = lasso(Xtrn{i},Ytrn,{i},'lambda',param);
        Wtmp1 = lasso(Xtrn{i},Ytrn{i},'lambda',0.001);
        ypred1 =  Xte{i}*Wtmp1;
        rmse1 = rsquare(ypred1,Yte{i});
        Wtmp2 = lasso(Xtrn{i},Ytrn{i},'lambda',0.01);
        ypred2 =  Xte{i}*Wtmp2;
        rmse2 = rsquare(ypred2,Yte{i});
        if rmse1<rmse2
            W(:,i) = Wtmp1;
            lambdaset(i) = 0.1;
        else
            W(:,i) = Wtmp2;
            lambdaset(i) = 0.01;
        end
    else
        no_trn_idx = [no_trn_idx,i];
    end
end
tmp = setdiff(1:length(Xtrn),no_trn_idx);
% W(:,no_trn_idx) = repmat(mean(W(:,tmp),2),1,length(no_trn_idx));
