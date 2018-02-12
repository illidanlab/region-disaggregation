function [W, G,Fval] = MTMLb(XL, Y, XR,rho1,rho2,rho3, opts)
R  = length (XL);
d = size(XL{1}, 2);
k = size(XR,2);
Fval = [];
iter = 0;
G = zeros(k,d);

while iter<opts.OutermaxIter
    % Learn W
    [W, f1] = MTMLb_2(XL, Y, XR, G,rho1,rho3, opts);
    % Learn G
    [G, f2] = MTMLb_1(XR, W, rho1,rho2, opts);
    
    % compute funcVal
    funcVal =0;
    for i = 1: R
        funcVal = funcVal + 0.5 * norm (Y{i} - XL{i} * W(:,i))^2 + ...
            rho1*0.5*norm(W(:,i)'-XR(i,:)*G)^2 ;
    end
    funcVal = funcVal + rho2*norm(G,1) + rho3*norm(W,1);
    
    lambda = -0.001;
%     lambda = -Inf;
    if iter >=1
        trend = (Fval(end)-funcVal)/Fval(end);
        if trend >= lambda
            Fval = cat(1,Fval,funcVal);
            %         elseif trend <=0.001 && trend >=0
            %              Fval = cat(1,Fval,funcVal);
            %              break;% stop and return current W,G
        elseif trend <lambda
            W = Wold;
            G = Gold;
            break; % stop and return previous X and W
        end
    else
        Fval = cat(1,Fval,funcVal);
    end
    
    Wold =W;
    Gold = G;
    iter = iter + 1;
end


