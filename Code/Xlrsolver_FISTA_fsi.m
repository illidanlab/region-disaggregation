 function [Xlr,fval] = Xlrsolver_FISTA_fsi(y,Xl,G,R,W0_old,Wl_old,Wr_old,Xr,Xlrold,lambda3,lambda4,maxiter,tol)
        %min_{X}||y - Xlr_{diag}w-diag(Xl_{diag}GXlr_{diag}^{T})||_{2}^{2} + 
        %\lambda_{3}\sum_{i<j}D_{i,j}||Xlr_{diag}PR_{i,j}||_{F}^{2} + \lambda_{4}||Xlr_{diag}P - X_{r}||_{F}^{2}
        r = length(y);
%         n = length(y{1});
        dr = size(Xlrold{1},2);
        yhat = cell(1,r);
        for region = 1:r
            yhat{region} = y{region}-W0_old(region) - Xl{region}*Wl_old(:,region);
        end

        [Xl, ~, ~, ~] = diagonalize(Xl, yhat);
        [Xlrold, samplesize, ~, yvect] = diagonalize(Xlrold, yhat);
        Xr_total = cell(1,r);
        for regionind = 1:r
            Xr_total{regionind} = repmat(Xr(regionind,:),[samplesize(regionind),1]);  
        end
        Xr = cat(1,Xr_total{:});
        G = repmat(G,[r,r]);
%         N = sum(samplesize);
        w = reshape(Wr_old,[size(Wr_old,1)*size(Wr_old,2),1]);
        p = eye(dr);
        p = repmat(p,[r,1]);
        %maxiter = 500;
        lr = 1;%initial learning rate
        fval = zeros(maxiter,1);
        yk = Xlrold;
%         D = exp(-D);
        fval(1) = primal_fval(yvect,Xl,Xlrold,w,G,lambda3,lambda4,R);
        for iter = 2:maxiter
            tk = 1;
            gXlr = grad_Xlr(yvect,Xl,G,w,Xr,yk,lambda3,lambda4);
            Xlrnew = Xlrold - lr*gXlr;
            fnew = primal_fval(yvect,Xl,Xlrnew,w,G,lambda3,lambda4,R);
            fold = primal_fval(yvect,Xl,Xlrold,w,G,lambda3,lambda4,R);
            for ls = 1:100 % line search
                   if(fold - fnew < (lr/2)*norm(gXlr,'fro').^2)
                        lr = lr * 0.5;
                        Xlrnew = Xlrold - lr*gXlr;
                        fnew = primal_fval(yvect,Xl,Xlrnew,w,G,lambda3,lambda4,R);
                   else
                       break;
                   end   
            end
            tk = sqrt(1+4*tk^2)/2 + 1/2;
            yk = Xlrnew + (tk-1)/(tk+1)*(Xlrnew-Xlrold);
            fval(iter,1) = fnew;
            Xlrold = Xlrnew;
            if abs(fval(iter)- fval(iter-1))/abs(fval(iter-1)) < tol
            break;
            end
        end
        
        Xlr = Xlrnew*p;
%% nested function
     %cluster grad
     function [grad] = clusgrad(Xlr)
         grad = 2*R*((Xlr*p)'*R)'*p';
     end
    %mean grad
     function [grad] = meangrad(Xlr,Xr)
         grad = 2*(Xlr*p-Xr)*p';
     end
    %funcval
    function [fval] = primal_fval(y,Xl,Xlr,Wr,G,lambda3,lambda4,R)
       test = (Xlr*p)'*R;
       clustloss = norm(test,'fro')^2;
       fval = norm(y-Xlr*Wr-diag(Xl*G*Xlr')).^2 + lambda3*clustloss + lambda4 * norm(Xlr*p-Xr,'fro').^2; 
    end

     function [gradXlr] = grad_Xlr(y,Xl,G,w,Xr,Xlrold,lambda3,lambda4)
            sqgrad = zeros(size(Xlrold));
            for ind = 1:length(y)
                sqgrad(ind,:) = 2*(Xlrold(ind,:)*w + Xl(ind,:)*G*Xlrold(ind,:)'-y(ind,:))*(w + (Xl(ind,:)*G)');
            end
            gradXlr = sqgrad + lambda3*clusgrad(Xlrold) + lambda4*meangrad(Xlrold,Xr);
     end
 end