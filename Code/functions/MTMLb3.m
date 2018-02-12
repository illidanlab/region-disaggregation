function [W, G,Fval] = MTMLb3(XL, Y, XR,rho1,rho2,rho3,opts)
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
    [W, f1] = MTMLb2_2(XL, Y, XR, G,rho1,rho3, opts,W);
    % Learn G
    [G, f2] = MTMLb_1(XR, W, rho1,rho2, opts,G);
    
    % compute funcVal
    funcVal =0;
    for i = 1: R
        funcVal = funcVal + 0.5 * norm (Y{i} - XL{i} * W(:,i))^2 + ...
            rho1*0.5*norm(W(:,i)'-XR(i,:)*G)^2 ;
    end
    %     funcVal = funcVal + rho2*norm(G,1) + rho3*norm(W,1);
    non_smooth_value = 0;
    for i = 1 : size(W, 1)
        w = W(i, :);
        non_smooth_value = non_smooth_value + norm(w, 2);
    end
    funcVal = funcVal+ rho2*sum(sum(abs(G)))+ rho3*non_smooth_value;
    Fval = cat(1,Fval,funcVal);
    
    %     if funcVal <Fval_best;
    %         G_best = G;
    %         W_best = W;
    %         Fval_best = funcVal;
    %     end
    
    %     if isfield(opts,'output') && opts.output
    %         output{iter,1} = W;
    %         output{iter,2} = G;
    %         output{iter,3} = f1(end);
    %         output{iter,4} = f2(end);
    %     end
    
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
