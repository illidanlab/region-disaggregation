function[loss] = meanloss(x)
    
    mu = zeros(5,size(x,2));
    mu = cvx(mu);
    muestimate = zeros(size(x));
    muestimate = cvx(muestimate);
    ind = 1;
    for region = 1:5
            mu(region,:)= sum(x(ind:ind+20-1,:),1)/20;
            muestimate(ind:ind+20-1,:)=repmat(mu(region,:),[20,1]);
            ind = ind + 20;
    end
    x = reshape(x,[size(x,1)*size(x,2),1]);
    muestimate = reshape(muestimate,[size(x,1)*size(x,2),1])
    loss = sum_square_abs(x-muestimate)/size(x,1);
end