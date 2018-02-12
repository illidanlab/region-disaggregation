    function Xlr = solve_Xlr_nor(y,Xl,D,G,Wl_old,Xr_bar_total,Xlrold,lambda3,lambda4,lambda5)
        %minimize sum||yhat - Xlr*Wlr|| + lambda3 * 1/D*
        %sum_{i<j}||Xlr_total(i,:) -Xlr_total(j,:)||_{2} + lambda4*||Xlr_total - Xr_bar_total||_{F}
        r = length(y);
        dr = size(Xlrold{1},2);
        n = length(y{1});
        cvx_begin quiet
        cvx_precision high
        variable Xlr(r*n, dr)
        minimize sqaureloss_nor(Xlr,y,Xl,G) + lambda3 * clusloss(Xlr,1./D) + lambda4 * varloss(Xlr,Xr_bar_total)+lambda5 * norm_nuc(Xlr)
%          + lambda5 * meanloss(Xlr)
%         meanloss(Xlr) >= 1;
        cvx_end
    end