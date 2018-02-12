function [norm_l1] = l1_mat(x)
    x = abs(x);
    norm_l1 = (sum(sum(x,1)));
end