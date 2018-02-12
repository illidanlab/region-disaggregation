function [W, G,Fval] = MTMLb2(XL, Y, XR,rho1,rho2,rho3,opts)
R  = length (XL);
d = size(XL{1}, 2);
k = size(XR,2);
Fval = [];
% Fval_best = Inf;
iter = 1;
G = zeros(k,d);
W = zeros(d,R);
while iter<opts.OutermaxIter
    % Learn W
    [W, f1] = MTMLb_2(XL, Y, XR, G,rho1,rho3, opts,W);
    % Learn G
    [G, f2] = MTMLb_1(XR, W, rho1,rho2, opts,G);
    
    % compute funcVal
    funcVal =0;
    for i = 1: R
        funcVal = funcVal + 0.5 * norm (Y{i} - XL{i} * W(:,i))^2 + ...
            rho1*0.5*norm(W(:,i)'-XR(i,:)*G)^2 ;
    end
    %     funcVal = funcVal + rho2*norm(G,1) + rho3*norm(W,1);
    funcVal = funcVal+ rho2*sum(sum(abs(G)))+ rho3*sum(sum(abs(W)));
    Fval = cat(1,Fval,funcVal);   
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


