function [Wl,Wr,G,Xlr,fval] =  MTML_L21_fs(y,Xl,Xr_bar,lambda1,lambda2,lambda3,lambda4,coords,G_init,maxiter,toler)
    %solving MTMLa with regional variable imputation
    %input:
    %   y: cell of n*1 vector; Xl: cell of n*dl matrix, Xr_bar: r*dr matrix, 
    %   Xlr: cell of n*dr matrix, lambda1:hyperparameter, lambda2:hyperparameter
    %   lambda3:hyperparameter coords: the cell of n*2 matrix, which
    %   contains the spatial coordinates
    %formulation: 
    %minimize: sum_{1=i}^{n}||y_{i}-Xl_{i}*Wl_{i}-
    %Xlr_{i}*Wr{i}-diag(Xl{i}*G*Xr{i}')|| + lambda1*||[Wl,Wr]||_{1}+ lambda2*||G||_{1} +
    %lambda3*sum_{i}sum_{j}d_{ij}||Xlr{i}-Xlr{j}||_{2}^{2} +
    %lambda4*||Xlr-Xr_bar}||_{F}^{2}
    
    
    %args
    
    [~, sampleN, ~, ~] = diagonalize(Xl, y);
    r = length(y);%number of regions
    dl = size(Xl{1},2);
    dr = size(Xr_bar,2);
    Xr_bar = (Xr_bar - repmat(mean(Xr_bar),[size(Xr_bar,1),1]))./repmat(std(Xr_bar),[size(Xr_bar,1),1]);
    %initialize the Xlr using the Xr_bar
    Xlr_init = cell(1,r);
    rng('default')
    for i = 1:r
        ni = sampleN(i);
        Xlr_init{i} = repmat(Xr_bar(i,:),[ni,1]);
%         + randn(ni,dr);
    end
    rng(2);
%     G_init = randn(dl,dr);
    coordG = cat(1,coords{:});
    D = squareform(pdist(coordG)); %distance matrix
    m = 1;
    for diagi = 1:r
        D(m:m+sampleN(diagi)-1,m:m+sampleN(diagi)-1) = D(m:m+sampleN(diagi)-1,m:m+sampleN(diagi)-1).*100;
        m = m + sampleN(diagi);
    end
%     Xr_bar_cell = Xlr_init;
    Xr_bar_total = cat(1,Xlr_init{:});%vertical concatenate regional mean value
    indR = 1;
    N = size(Xr_bar_total,1);
    ivec = zeros(N*(N-1)/2,1);
    jvec = zeros(N*(N-1)/2,1);
    vvec = zeros(N*(N-1)/2,1);
    DR = exp(-D);
    for Rrow = 1:N-1
        for Rcol = Rrow+1:N
            ivec(indR,1) = Rrow;
            jvec(indR,1) = indR;
            vvec(indR,1) = sqrt(DR(Rrow,Rcol));
            ivec(indR+1,1) = Rcol;
            jvec(indR+1,1) = indR;
            vvec(indR+1,1) = -sqrt(DR(Rrow,Rcol));
            indR = indR + 1;  
            
        end 
    end
    
    R = sparse(ivec,jvec,vvec,size(Xr_bar_total,1),size(Xr_bar_total,1)*(size(Xr_bar_total,1)-1)/2);
%     R = sparse(double(R));
    Xlr_old = Xlr_init;
    G_old = G_init;
    fval = zeros(maxiter,1);
    fval(1) = primal_fval(y,Xl,Xlr_old,cat(1,Xlr_init{:}),Xr_bar_total,rand(dl,r),rand(dr,r),G_old,lambda1,lambda2,lambda3,lambda4,D);
    %BCD starts here
    for i = 2:maxiter
        %fix Xlr, G, solving Wl,Wr;
        [Wl,Wr] = solve_WlWr(y,Xl,Xlr_old,G_old,lambda1);
        %fix Wl,Wr,Xlr, solving G;
        Wl_old = Wl;
        Wr_old = Wr;
        G = solve_G(y,Xl,Xlr_old,Wl_old,Wr_old,lambda2);
        [Xlr_total,~] = Xlrsolver_FISTA_fs(y,Xl,G,R,Wl_old,Wr_old,Xr_bar,Xlr_old,lambda3,lambda4,500,0.01);
        Xlr_old = cell(1,r);
        ind = 1;
        for j = 1:r
            Xlr_old{j} = Xlr_total(ind:ind+sampleN(j) -1,:);
            ind = ind + sampleN(j);
        end
        G_old = G;
        fval(i,1) = primal_fval(y,Xl,Xlr_old,Xlr_total,Xr_bar_total,Wl,Wr,G,lambda1,lambda2,lambda3,lambda4,D);
        if abs(fval(i)- fval(i-1))/abs(fval(i-1)) < toler
            break;
        end        
    end
    Xlr = Xlr_old;
    
% nested function
    %% 
    function [fval] = primal_fval(y,Xl,Xlr,Xlr_total,Xr_bar_total,Wl,Wr,G,lambda1,lambda2,lambda3,lambda4,D)
        fval = 0;
        %clustering loss
%         conttable = zeros(50*49/2,1)
        contRegu = norm((Xlr_total)'*R,'fro')^2;
        discRegu = norm(Xr_bar_total-Xlr_total,'fro')^2;%loss of variance within one region
        for region = 1:r
            rmse = norm(y{region}-Xl{region}*Wl(:,region) - Xlr{region}*Wr(:,region)-diag(Xl{region}*G*Xlr{region}'))^2;
            fval = fval + rmse; 
        end
        
        fval = fval + lambda1 * sum(sqrt(sum(vertcat(Wl,Wr).^2, 2))) + lambda2 * l1_mat(G) + lambda3 * contRegu ...,
                    + lambda4 * discRegu;
    end

    function [Wl,Wr] = solve_WlWr(y,Xl,Xlr_old,G_old,lambda1)
        %minimize sum_||yi-Xl*G*Xlr'-[Xl,Xlr]*[Wl(:,i),Wr(:,i)]|| +
        %lambda1|[Xl,Xlr]|_{1}
        yhat = cell(1,r);
        X_all = cell(1,r);
        for region = 1:r
            yhat{region} = y{region} - diag(Xl{region}*G_old*Xlr_old{region}');
            X_all{region} = horzcat(Xl{region},Xlr_old{region});
        end
        W = Least_L21(X_all,yhat,lambda1);
        Wl = W(1:dl,:);
        Wr = W(dl+1:end,:);
    end
    
    function G = solve_G(y,Xl,Xlr_old,Wl_old,Wr_old,lambda2)
        %solve global regression for G
        y_all = cell(1,r);
        for region = 1:r
            y_all{region} = y{region} - Xl{region}*Wl_old(:,region) - Xlr_old{region}*Wr_old(:,region);
        end
        y_all = cat(1,y{:});
        X_cross = createX_csi_local(Xl,Xlr_old);
        X_all = cat(1,X_cross{:});
        G_flat = lasso(X_all,y_all,'Lambda',lambda2);
        G = reshape(G_flat,[dr,dl])';
    end
end