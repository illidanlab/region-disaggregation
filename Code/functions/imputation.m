function [Xlr,fval] =  imputation(y,Xl,Xr_bar,lambda3,lambda4,lambda5,coords)
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
    maxiter =3;
    r = length(y);%number of regions
    dl = size(Xl{1},2);
    dr = size(Xr_bar,2);
    Xr_bar = (Xr_bar - repmat(mean(Xr_bar),[size(Xr_bar,1),1]))./repmat(std(Xr_bar),[size(Xr_bar,1),1])
    %initialize the Xlr using the Xr_bar
    Xlr_init = cell(1,r);
    for i = 1:r
        ni = size(Xl{i},1);
        Xlr_init{i} = repmat(Xr_bar(i,:),[ni,1]);
    end
    coordG = cat(1,coords{:});
    D = squareform(pdist(coordG,'mahalanobis')); %distance matrix
    
%     Xr_bar_cell = Xlr_init;
    Xr_bar_total = cat(1,Xlr_init{:});%vertical concatenate regional mean value
    Xlr_old = Xlr_init;
    fval = zeros(maxiter,1);
    %BCD starts here
    for i = 1:maxiter
        Xlr_total = impute_Xlr(D,Xr_bar_total,Xlr_old,lambda3,lambda4,lambda5);
        Xlr_old = cell(1,r);
        ind = 1;
        for j = 1:r
            Xlr_old{j} = Xlr_total(ind:ind+size(Xlr_total,1)/r -1,:);
            ind = ind + size(Xlr_total,1)/r;
        end
       
        fval(i,1) = primal_fval(Xlr_old,Xlr_total,Xr_bar_total,lambda3,lambda4,D);
        i
    end
    Xlr = Xlr_old;
    
% nested function
    %% 
    function [fval] = primal_fval(Xlr,Xlr_total,Xr_bar_total,lambda3,lambda4,D)
       
        fval = 0;
        contRegu = 0;%clustering loss
%         conttable = zeros(50*49/2,1)
        for indexi = 1:(size(Xlr,1)-1) %loop for every data point
            for indexj = (indexi+1) : size(Xlr,1)
                contRegu = contRegu + (1./D(indexi,indexj))*(norm(Xlr_total(indexi,:)-Xlr_total(indexj,:))^2);
%                 conttable(indexi,1) = (1./D(indexi,indexj))*(norm(Xlr(indexi,:)-Xlr(indexj,:))^2);
            end
        end
        discRegu = norm(Xr_bar_total-Xlr_total,'fro')^2;%loss of variance within one region
      
        for region = 1:r
            fval = fval + lambda3 * contRegu ...,
                    + lambda4 * discRegu;
        end  
    end

   
% 
%     function Xlr = solve_Xlr(y,Xl,D,G,Wl_old,Wr_old,Xr_bar_total,Xlrold,lambda3,lambda4)
%         %minimize sum||yhat - Xlr*Wlr|| + lambda3 * 1/D*
%         %sum_{i<j}||Xlr_total(i,:) -Xlr_total(j,:)||_{2} + lambda4*||Xlr_total - Xr_bar_total||_{F}
%         yhat = cell(1,r);
%         for region = 1:r
%             yhat{region} = y{region} - Xl{region}*Wl_old(:,region) -diag(Xl{region}*G*Xlrold{i}');
%         end
%         cvx_begin quiet
%         cvx_precision high
%         variable Xlr(r*n, dr)
%         minimize sqaureloss(Xlr,y,Wr_old) + clusloss(Xlr,Xr_bar_total,1./D) + varloss(Xlr,Xr_bar_total)
%         cvx_end
%         Xlr = Xlrold;
%     end
    
end