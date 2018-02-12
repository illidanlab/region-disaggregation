function [U,V,RR,Fval,W,output] = MTMLc2_test(XL, Y, XR,m,rho1,rho2,rho3,rho4,opts)
R  = length (XL);
% k = size(XR,2);
d = size(XL{1},2);
Fval = [];
% Fval_best = Inf;
iter = 1;
testfuncval = -Inf;
% randn('state',2016);
% rand('state',2016);
% initalization
if isfield(opts,'initW')
    [U,V] = nnmf(opts.initW,m,'replicates',100);
else
    U = randn(d,m);
    V = randn(m,R);
end
[RR,~] = MTMLc_3(XR, V, rho1,rho4, opts);

while iter<=opts.OutermaxIter
    Vcur = V;
    Ucur = U;
    RRcur = RR;
    % Learn V
    [V, f2] = MTMLc_2(XL, Y, XR, U, RR,rho1,rho3, opts,Vcur);
    [funcVal] = funVal_eval(Ucur,V,RRcur); Fval = cat(1,Fval,funcVal);
    if(iter>1)
        testfuncval = Fval(end)-Fval(end-1);
    end
    % Learn U
    [U, f1] = MTMLc_1(XL, Y, V', rho2, opts, Ucur');
    U = U';
    [funcVal] = funVal_eval(U,V,RRcur);Fval = cat(1,Fval,funcVal);
    if(iter>1)
        testfuncval = Fval(end)-Fval(end-1);
    end
    % Learn RR
    [RR,f3] = MTMLc_3(XR, V, rho1,rho4, opts, RRcur);
    [funcVal] = funVal_eval(U,V,RR);Fval = cat(1,Fval,funcVal);
    if(iter>1)
        testfuncval = Fval(end)-Fval(end-1);
    end
    
    if isfield(opts,'output') && opts.output
        output{iter,1} = U;
        output{iter,2} = V;
        output{iter,3} = RR;
        output{iter,4} = f1(end);
        output{iter,5} = f2(end);
        output{iter,6} = f3(end);
    end
    %     % check stopping criteria
    %     if iter >=2
    %         if abs(Fval(end-1)-funcVal)<= Fval(end-1)*opts.tol;
    %             if(opts.verbose)
    %                 fprintf('\n The program terminates as the relative change of funcVal is small. \n');
    %             end
    %             break;
    %         end
    %     end
    
    iter = iter + 1;
end
W = U*V;

    function [funcVal] = funVal_eval(U,V,RR)
        funcVal = 0;
        for ii = 1: R
            funcVal = funcVal + 0.5 * norm (Y{ii} - XL{ii} * U*V(:,ii))^2;
        end
        %         funcVal = funcVal + rho1*0.5*norm(XR'-RR*V,'fro')^2+ ...
        %             rho2*norm(U,1) + rho3*norm(V,1)+rho4*norm(RR,1);
        funcVal = funcVal + rho1*0.5*norm(XR'-RR*V,'fro')^2+ ...
            rho2*sum(sum(abs(U))) + rho3*sum(sum(abs(V)))+rho4*sum(sum(abs(RR)));
    end
end


