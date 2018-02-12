function [U,V,RR,Fval,W] = MTMLc(XL, Y, XR,m,rho1,rho2,rho3,rho4,opts)
R  = length (XL);
k = size(XR,2);
Fval = [];
iter = 0;
randn('state',2016);
rand('state',2016);
V = randn(m,R);
RR = randn(k,m);

while iter<opts.OutermaxIter
    % Learn U
    [U, ~] = MTMLc_1(XL, Y, V', rho2, opts);
    U = U';
    % Learn V
    [V, ~] = MTMLc_2(XL, Y, XR, U, RR,rho1,rho3, opts);
    % Learn RR
    [RR,~] = MTMLc_3(XR, V, rho1,rho4, opts);
    
    % compute funcVal
    funcVal =0;
    for i = 1: R
        funcVal = funcVal + 0.5 * norm (Y{i} - XL{i} * U*V(:,i))^2;
    end
    funcVal = funcVal + rho1*0.5*norm(XR'-RR*V,'fro')^2+ ...
        rho2*norm(U,1) + rho3*norm(V,1)+rho4*norm(RR,1);
    
%     lambda = -0.001;
lambda = -Inf;
     if iter >=1
        trend = (Fval(end)-funcVal)/Fval(end);
        if trend > lambda
             Fval = cat(1,Fval,funcVal);
%         elseif trend <=0.01 && trend >0
%              Fval = cat(1,Fval,funcVal);
% %              break;% stop and return current W,G
        elseif trend <=lambda
            U = Uold;
            V = Vold;
            RR = RRold;
            break; % stop and return previous X and W
        end
    else
        Fval = cat(1,Fval,funcVal);
     end
    
    Uold = U;
    Vold = V;
    RRold = RR;
    iter = iter + 1;
end
W = U*V;


