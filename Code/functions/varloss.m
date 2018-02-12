function [loss] = varloss(x,xmean)
    length = size(x,1)*size(x,2);
    x = reshape(x,[length,1]);
    xmean = reshape(xmean,[length,1]);
    loss = sum_square_abs(x-xmean);
end