    function Xlr = impute_Xlr(D,Xr_bar_total,Xlrold,lambda3,lambda4,lambda5)
        %minimize sum||yhat - Xlr*Wlr|| + lambda3 * 1/D*
        %sum_{i<j}||Xlr_total(i,:) -Xlr_total(j,:)||_{2} + lambda4*||Xlr_total - Xr_bar_total||_{F}
        r = length(Xlrold);
        n = length(Xlrold{1});
        dr = size(Xlrold{1},2);
        cvx_begin quiet
        cvx_precision high
        variable Xlr(r*n, dr)
        minimize lambda3 * clusloss(Xlr,1./D) + lambda4 * varloss(Xlr,Xr_bar_total)+lambda5 * norm_nuc(Xlr)
%          + lambda5 * meanloss(Xlr)
%         meanloss(Xlr) >= 1;
        cvx_end
    end