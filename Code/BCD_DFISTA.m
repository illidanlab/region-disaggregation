function [ W, info, Th, q ] = BCD_DFISTA(  X, y, lambda1, lambda2, lambda3, X_csi, opts)
%
% Multi-Task Multi-Level Feature Learning with Calibration
% diagnoal version for faster computation on small sample size.
% dual projected gradient -- using solver. 
%
% OBJECTIVE
%    min_W { 1/2sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2 +lambda3/2 ||q||_2^{2} }
%
%  We solve this by the dual form
%    max_Theta {min_W { sum_i^m theta_i^T (Xi wi - yi) + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2 }
%              +min_q {lambda3/2||q||^{2} - sum_i^{m} theta_i^T x_i^CSI q}}
%          s.t.,     ||theta_i|| <= sqrt(2)/2 (i = 1..m)
%
%
% INPUT
%  X - cell array of {n_i by d matrices} by m
%  y - cell array of {n_i by 1 vectors}  by m
%  x_CSI - cell array of (n_i by d_CSI matrics) by m
%  lambda1 - regularization parameter of the l2,1 norm penalty for w
%  lambda2 - regularization parameter of the Fro norm penalty for w
%  lambda3 - regularization parameter of the l2 norm penalty for q
% OUTPUT
%  W - task weights: d by t.
%  q - CSI: d_CSI by 1
%  funcVal - the funcion value.
%
% Author: Jiayu Zhou, Pinghua Gong

%% Initialization
% mex segL2Proj.c
if(nargin<7), opts = []; end

opts = setOptsDefault( opts, 'verbose', 1); 
opts = setOptsDefault( opts, 'maxIter', 100);
opts = setOptsDefault( opts, 'tol',     1e-5);
opts = setOptsDefault( opts, 'stopflag', 1);
verbose = opts.verbose;

globalVerbose = 1;
globalMaxIter = 10000;
globalTol     = 1e-9;

info.algName = 'Dual MLMT FISTA';

% if verbose > 0
%     fprintf('%s: Config [MaxIter %u][Tol %.4g]\n', info.algName, opts.maxIter, opts.tol);
% end

r = length(X); % task number

d = size(X{1}, 2);
d_csi = size(X_csi{1},2);

% diagonalize X and vectorized y.
[Xdiag, samplesize, Th_vecIdx, yvect] = diagonalize(X, y);
[X_csi_diag, samplesize, Th_vecIdx, yvect] = diagonalize(X_csi, y);

info.fvP = zeros(opts.maxIter, 1);
timeVal  = zeros(opts.maxIter, 1);
timeP = zeros(opts.maxIter, 1);
timeQ = zeros(opts.maxIter,1);



if isfield(opts, 'initTheta')
    Th0 = segL2Proj(opts.initTheta, Th_vecIdx)/2;
%     if verbose > 0, fprintf('%s: use given initial point.\n', info.algName), end
else
    Th0 = (segL2Proj(randn(sum(samplesize), 1), Th_vecIdx))/2;
end

%% Computation
% if verbose == 1; fprintf('Iteration:     '); end

% bFlag = 0; % whether the gradient step only changes little.

Thk     = Th0;
for iter = 1: opts.maxIter
    iterTic = tic;
    %Fix q, solve p
    W = computeW(Thk,r,d);
    q = computeQ(Thk,lambda3,X_csi_diag,d_csi,r);
    y_W = cell(5,1);
    for i = 1:r
        y_W{i} = y{i} - X_csi{i}*q;
    end
    opts_DFISTA = [];
    opts_DFISTA.verbose = globalVerbose;
    opts_DFISTA.maxIter = iter*10;
    opts_DFISTA.tol     = globalTol;
    opts_DFISTA.initTheta = Thk;
    [W_new,info_W,theta_w] = MTFLCd_DFISTA( X, y_W, lambda1, lambda2, opts_DFISTA );
    timeP(iter) = sum(info_W.timeVal);

    %Fix p, solve q
    y_q = cell(5,1);
    for i = 1:r
        y_q{i} = y{i} - X{i}*W(:,i);
    end
    [q_new, info_q, theta_q] = Dual_Qsolver( X_csi, y_W, lambda3, opts_DFISTA,theta_w);
    timeQ(iter) = sum(info_q.timeVal);
    Thk = theta_q;
    info.fvP(iter) = primalObjective(W_new,q_new, X, y,X_csi, lambda1, lambda2,lambda3);
    % test stop condition.
 %   if (bFlag), break; end
    if iter > 1, timeVal(iter) = timeVal(iter-1) + toc(iterTic);
    else timeVal(iter) = toc(iterTic); end
    if iter>=2
        if (abs( info.fvP(iter) - info.fvP(iter-1) ) <= opts.tol* abs(info.fvP(iter-1)))
             break;
        end
        
    end
end
% if verbose == 1; fprintf('\n'); end

%% Output.
W   = W_new;
Th  = Thk;
q = q_new;
info.fvP = info.fvP(1:iter);
info.timeVal = timeVal (1:iter);
info.timeQ = timeQ(1:iter);
info.timeP = timeP(1:iter);



%% Nested Functions
    function fvP = primalObjective(W,q, X, y,X_csi, lambda1, lambda2,lambda3)
        fv = lambda1 * sum(sqrt(sum(W.^2, 2))) + lambda2 /2 * sum(sum(W.^2)) + lambda3/2*(sqrt(sum(q.^2)));
        for i = 1: length(X)
            fvP = fv + sqrt(sum((X{i} * W(:, i) + X_csi{i}*q - y{i}).^2));
        end
    end

    function WTh = computeW(Th_vec,r,d)
        WTh = reshape(Th_vec' * Xdiag, d, r);% Compute UTh
        WTh = -1/lambda2*max(0,1-lambda1./repmat(sqrt(sum(WTh.^2,2)),1,r)).*WTh;
    end

    function q = computeQ(Th_vec,lambda3,X_csi_diag,d_csi,r)
        TH0_mat = reshape(Th_vec'*X_csi_diag,d_csi,r);
        eta = sum(TH0_mat,2);
        q = -eta/lambda3;
    end
end