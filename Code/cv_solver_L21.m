function [Wl,gamma] =  cv_solver_L21(y,Xl,Xlr,lambda1,lambda2,G_init,maxiter,toler)
    %solving MTMLa with regional variable imputation
    %input:
    %args
    
%     [~, sampleN, ~, ~] = diagonalize(Xl, y);
    r = length(y);%number of regions
    dl = size(Xl{1},2);
    dr = size(Xlr{1},2);
    rng(2);
    GestimateBi = double(G_init~=0);
    gammabi = reshape(GestimateBi',[dl*dr,1]);
    indnzero = find(gammabi~=0);
    Xlrcsi = createX_csi_local(Xl,Xlr);
    for i = 1:r
        Xlrcsi{i} = Xlrcsi{i}(:,indnzero);
    end
    Xltmp = cell(1,r);
    for i = 1:r
        Xltmp{i} = horzcat(Xl{i},Xlr{i});
    end
    fval = zeros(maxiter,1);
    Wlinit = randn(dl+dr,r);
    gamma_init = randn(size(Xlrcsi{1},2),1);
    fval(1) = primal_fval(y,Xltmp,Xlrcsi,Wlinit,gamma_init,lambda1,lambda2);
    %BCD starts here
    gamma_old = gamma_init;
    for i = 2:maxiter
        %fix Xlr, G, solving Wl,Wr;
        [Wl] = solve_WlWr(y,Xltmp,Xlrcsi,gamma_old,lambda1);
        %fix Wl,Wr,Xlr, solving G;
        Wl_old = Wl;
        gamma = solve_G(y,Xltmp,Xlrcsi,Wl_old,lambda2);
        gamma_old = gamma;
        fval(i,1) = primal_fval(y,Xltmp,Xlrcsi,Wl,gamma,lambda1,lambda2);
        if abs(fval(i)- fval(i-1))/abs(fval(i-1)) < toler
            break;
        end        
    end
%     fval
    
% nested function
    %% 
    function [fval] = primal_fval(y,Xl,Xlrcsi,Wl,gamma,lambda1,lambda2)
        fval = 0;
        
        for region = 1:r
            rmse = norm(y{region}-Xl{region}*Wl(:,region) - Xlrcsi{region}*gamma);
            
        end
        fval = fval + rmse + lambda1 * sum(sqrt(sum(Wl.^2, 2))) + lambda2 * l1_mat(gamma);
      
    end

    function [W] = solve_WlWr(y,Xl,Xlr,G_old,lambda1)
        %minimize sum_||yi-Xl*G*Xlr'-[Xl,Xlr]*[Wl(:,i),Wr(:,i)]|| +
        %lambda1|[Xl,Xlr]|_{1}
        yhat = cell(1,r);
        
        for region = 1:r
            yhat{region} = y{region} - Xlr{region}*G_old;
        end
        W = Least_L21(Xl,yhat,lambda1);
        
    end
    
    function gamma = solve_G(y,Xl,Xlrcsi,Wl_old,lambda2)
        %solve global regression for G
        y_all = cell(1,r);
        for region = 1:r
            y_all{region} = y{region} - Xl{region}*Wl_old(:,region);
        end
        y_all = cat(1,y_all{:});
        X_all = cat(1,Xlrcsi{:});
        gamma = lasso(X_all,y_all,'Lambda',lambda2);
%         G = reshape(G_flat,[dr,dl])';
    end
end