    function Xlr = solve_Xlr(y,Xl,D,G,Wl_old,Wr_old,Xr_bar_total,Xlrold,lambda3,lambda4,lambda5)
        %minimize sum||yhat - Xlr*Wlr|| + lambda3 * 1/D*
        %sum_{i<j}||Xlr_total(i,:) -Xlr_total(j,:)||_{2} + lambda4*||Xlr_total - Xr_bar_total||_{F}
        r = length(y);
        n = length(y{1});
        dr = size(Xlrold{1},2);
        yhat = cell(1,r);
        for region = 1:r
            yhat{region} = y{region} - Xl{region}*Wl_old(:,region);
        end
        cvx_begin quiet
        cvx_precision high
        variable Xlr(r*n, dr)
        minimize sqaureloss(Xlr,y,Wr_old,Xl,G) + lambda3 * clusloss(Xlr,exp(-D)) + lambda4 * varloss(Xlr,Xr_bar_total)+lambda5 * norm_nuc(Xlr)
%          + lambda5 * meanloss(Xlr)
%         meanloss(Xlr) >= 1;
        cvx_end
        
        
%         Xlr_old = cell(1,r);
%         ind = 1;
%         for j = 1:r
%             Xlr_old{j} = Xlr(ind:ind+size(Xlr,1)/r -1,:);
%             ind = ind + size(Xlr,1)/r;
%         end
%         xlrcvx = Xlr;
%         [Xl, samplesize, accIdx, yvect] = diagonalize(Xl, y);
%         [Xlrold, samplesize, accIdx, yvect] = diagonalize(Xlr_old, y);
% %         Xr_total = cell(1,r);
% %         for regionind = 1:r
% %             Xr_total{regionind} = repmat(Xr(regionind,:),[samplesize(regionind),1]);  
% %         end
% %         Xr = cat(1,Xr_total{:});
%         G = repmat(G,[r,r]);
%         N = sum(samplesize);
%         w = reshape(Wr_old,[size(Wr_old,1)*size(Wr_old,2),1]);
% %         p = eye(dr);
% %         p = repmat(p,[r,1]);
%         %maxiter = 500
%         norm(yvect-Xlrold*w-diag(Xl*G*Xlrold')).^2
%         [Xlr_total,f] = Xlrsolver_FISTA(y,Xl,D,G,Wl_old,Wr_old,Xr_bar_total,Xlr_old,lambda3,lambda4,2,1e-12);
    end