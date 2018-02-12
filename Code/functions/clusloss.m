function[loss] = clusloss(x,D)
    loss = 0;
    for i = 1:size(x,1)-1
        for j = i+1:size(x,1)
            loss = loss + D(i,j) * sum_square_abs(x(i,:)-x(j,:));
        end
    end
end