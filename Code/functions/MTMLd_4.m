function W = MTMLd_4(XL, Y, Gamma, tau, U, V,rho2,opts,initW)
% W is the learned Z
if nargin <8
    opts = [];
end

% initialize options.
opts=init_opts(opts);
if ~isfield(opts, 'verbose') % print every iteration
    opts.verbose = 0;
end

r  = length (XL);
d = size(XL{1},2);

funcVal = [];
if nargin == 9
    W0 = initW;
else
    W0 = zeros( d ,r );
end

bFlag=0; % this flag tests whether the gradient step only changes a little

Wz= W0;
Wz_old = W0;
t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
   
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval  (Ws);
    
    % line search
    while true
        Wzp = FGLasso_projection(Ws - gWs/gamma, rho2 / gamma);
        Fzp = funVal_eval (Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;
        %         Fzp_gamma = Fs + trace(delta_Wzp' * gWs) + gamma/2 * norm(delta_Wzp, 'fro')^2;
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs)) + gamma/2 * norm(delta_Wzp, 'fro')^2;% eq(7)
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            %         if (Fzp -Fzp_gamma < 0 || abs(Fzp-Fzp_gamma)<1e-5 )
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    %     funcVal = cat(1, funcVal, Fzp + rho3 * l1c_wzp);
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho2));
    if (bFlag)
        if(opts.verbose)
            fprintf('\n The program terminates as the gradient step changes the solution(W) very small. \n');
        end
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    if(opts.verbose)
                        fprintf('\n The program terminates as the absolute change of funcVal is small. \n');
                    end
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    if(opts.verbose)
                        fprintf('\n The program terminates as the relative change of funcVal is small. \n');
                    end
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                if(opts.verbose)
                    fprintf('\n The program terminates as the funcVal is lower than threshold. \n');
                end
                break;
            end
        case 3
            if iter>=opts.maxIter
                if(opts.verbose)
                    fprintf('\n The program terminates as it reaches the maximum iteration. \n');
                end
                break;
            end
    end
    
    if(opts.verbose)
        fprintf('Iteration %8i| function value %12.4f \n',iter,funcVal(end));
    end
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end
W = Wzp;


% private functions
    function [grad_W] = gradVal_eval(Z)
        if opts.pFlag
            grad_W = zeros(size(Z));
            parfor t_ii = 1:r
                grad_W(:,t_ii) = - (XL{t_ii})'*(Y{t_ii}-XL{t_ii}*Z(:,t_ii));
            end
            grad_W = grad_W + 2*tau*(Z-U*V);
        else
            grad_W = zeros(size(Z));
            for t_ii = 1:r
                grad_W(:,t_ii) = - (XL{t_ii})'*(Y{t_ii}-XL{t_ii}*Z(:,t_ii));
            end
            grad_W = grad_W + 2*tau*(Z-U*V)+Gamma;
        end
    end

    function [funcVal] = funVal_eval (Z)
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: r
                funcVal = funcVal + 0.5 * norm (Y{i} - XL{i} * Z(:,i))^2 + Gamma(:,i)'*(Z(:,i) - U*V(:,i));
            end
            funcVal = funcVal + tau*norm(Z-U*V,'fro');
        else
            for i = 1: r
                funcVal = funcVal + 0.5 * norm (Y{i} - XL{i} * Z(:,i))^2 + Gamma(:,i)'*(Z(:,i) - U*V(:,i));
            end
            funcVal = funcVal + tau*norm(Z-U*V,'fro');
        end
    end

    function [Wp] = FGLasso_projection (W, lambda )
        % solve it in row wise (L_{2,1} is row coupled).
        % for each row we need to solve the proximal opterator
        % argmin_w { 0.5 \|w - v\|_2^2 + lambda_3 * \|w\|_2 }
        
        Wp = zeros(size(W));
        
        if opts.pFlag
            parfor i = 1 : size(W, 1)
                v = W(i, :);
                nm = norm(v, 2);
                if nm == 0
                    w = zeros(size(v));
                else
                    w = max(nm - lambda, 0)/nm * v;
                end
                Wp(i, :) = w';
            end
        else
            for i = 1 : size(W, 1)
                v = W(i, :);
                nm = norm(v, 2);
                if nm == 0
                    w = zeros(size(v));
                else
                    w = max(nm - lambda, 0)/nm * v;
                end
                Wp(i, :) = w';
            end
        end
    end

    function [non_smooth_value] = nonsmooth_eval(W, rho_1)
        non_smooth_value = 0;
        if opts.pFlag
            parfor i = 1 : size(W, 1)
                w = W(i, :);
                non_smooth_value = non_smooth_value ...
                    + rho_1 * norm(w, 2);
            end
        else
            for i = 1 : size(W, 1)
                w = W(i, :);
                non_smooth_value = non_smooth_value ...
                    + rho_1 * norm(w, 2);
            end
        end
    end

end