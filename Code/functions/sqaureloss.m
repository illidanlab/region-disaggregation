function [loss] = sqaureloss(xlr,y,w,xl,g)
    r = length(y);
    n = size(xl{1},1);
    loss = 0;
    ind = 1;
    for i = 1:r
        loss = loss + sum_square_abs(y{i}-xlr(ind:ind+n-1,:)*w(:,i) - diag(xl{i}*g*xlr(ind:ind+n-1,:)'));
        ind = ind+n;
    end
end