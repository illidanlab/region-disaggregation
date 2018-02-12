
cvx_begin quiet
cvx_precision high
    
variable G(13, 8)
minimize(norm(W - G*X_R','fro'))
cvx_end

function [ fv ] = primalObj_cvx(W,G,Z)
% CVX: primal objective
fv = norm(W - G*Z','fro')
end
