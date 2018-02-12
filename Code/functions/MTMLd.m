function [U,V,R,Z,Gamma,Fval,W] = MTMLd(XL, Y, XR,m,rho1,rho2,opts)
% Input:
% XL r by 1 cell, XL{i} is the preditor of ith region
% Y r by 1 cell, Y{i} is the repnose vector of ith region
% XR k by r, regional predictor, dimension k
 
r  = length (XL);
k = size(XR,2);
d = size(XL{1},2);
Fval = [];

iter = 1;
randn('state',2016);
rand('state',2016);

% Initalization
% if isfield(opts,'initW')
%     [UU,SS,VV] = svds(opts.initW,m);
%     U = UU*sqrt(SS);
%     V = sqrt(SS)*VV';
% %     [U,V] = nnmf(opts.initW,m,'replicates',100);
% else
%     U = randn(d,m);
%     V = randn(m,r);
% end
V = randn(m,r);
R = randn(k,m);
Z = randn(d,r);
Gamma = zeros(d,r);
tau = opts.tau;
while iter<=opts.OutermaxIter
    
    % Learn U
    U = MTMLd_1(Gamma, V, Z,tau);
    
    % Learn V
    V = MTMLd_2(R,U,XR,Z,Gamma, rho1,tau);
    
    % Learn R
    R = MTMLd_3(XR, V);
    
    % Learn Z
    Z = MTMLd_4(XL, Y, Gamma, tau, U, V,rho2,opts,Z);
    
    % Update Gamma
    Gamma = Gamma + tau*(Z-U*V);
    
    % Update objective funciton
    funcVal = funVal_eval(U*V);
    Fval = cat(1,Fval,funcVal);
    
    % check stopping criteria
    if iter >=2
        if abs(Fval(end-1)-funcVal)<= Fval(end-1)*opts.tol;
            if(opts.verbose)
                fprintf('\n The program terminates as the relative change of funcVal is small. \n');
            end
            break;
        end
    end
    
    iter = iter + 1;
end

W = U*V;

    function [funcVal] = funVal_eval(W)
        funcVal = 0;
        for ii = 1: r
            funcVal = funcVal + 0.5 * norm (Y{ii} - XL{ii} * U*V(:,ii))^2;
        end
        funcVal = funcVal + rho1*0.5*norm(XR'-R*V,'fro')^2;
        non_smooth_value = 0;
        for i = 1 : d
            w = W(i, :);
            non_smooth_value = non_smooth_value + rho2 * norm(w, 2);
        end
        funcVal = funcVal + non_smooth_value;
        
    end

end


