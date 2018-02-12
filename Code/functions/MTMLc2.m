function [U,V,RR,Fval,W] = MTMLc2(XL, Y, XR,m,rho1,rho2,rho3,rho4,opts)
R  = length (XL);
% k = size(XR,2);
d = size(XL{1},2);
Fval = [];
% Fval_best = Inf;
iter = 1;
% randn('state',2016);
% rand('state',2016);
% initalization
if isfield(opts,'initW')
    [UU,SS,VV] = svds(opts.initW,m);
    U = UU*sqrt(SS);
    V = sqrt(SS)*VV';
%     [U,V] = nnmf(opts.initW,m,'replicates',100);
else
    U = randn(d,m);
    V = randn(m,R);
end
[RR,~] = MTMLc_3(XR, V, rho1,rho4, opts);

while iter<=opts.OutermaxIter
    % Learn V
    [V, ~] = MTMLc_2(XL, Y, XR, U, RR,rho1,rho3, opts,V);
%     f2 = funVal_eval();
    % Learn U
    [U, ~] = MTMLc_1(XL, Y, V', rho2, opts, U');
    U = U';
%     f1 = funVal_eval();
    % Learn RR
    [RR,~] = MTMLc_3(XR, V, rho1,rho4, opts, RR);
    funcVal = funVal_eval();
    Fval = cat(1,Fval,funcVal);
    
%     if isfield(opts,'output') && opts.output
%         output{iter,1} = U;
%         output{iter,2} = V;
%         output{iter,3} = RR;
%         output{iter,4} = f1;
%         output{iter,5} = f2;
%         output{iter,6} = funcVal;
%     end

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

    function [funcVal] = funVal_eval()
        funcVal = 0;
        for ii = 1: R
            funcVal = funcVal + 0.5 * norm (Y{ii} - XL{ii} * U*V(:,ii))^2;
        end
        funcVal = funcVal + rho1*0.5*norm(XR'-RR*V,'fro')^2+ ...
            rho2*sum(sum(abs(U))) + rho3*sum(sum(abs(V)))+rho4*sum(sum(abs(RR)));
    end
end


