function[rsquare, rmse] = CalcError4Calibration(P,q,X, X_csi,y)
    numofTask = length(X);
    y_pred = cell(numofTask,1);
% y_hat{1}=rand(200,1);y_hat{2}=rand(200,1);y_hat{3}=rand(200,1);y_hat{4}=rand(200,1);y_hat{5}=rand(200,1);
    for i = 1:size(y,1)
        y_pred{i} = X{i}*P(:,i)+X_csi{i}*q;
    end
    [testttt, testt, Th_vecIdx, y_vec] = diagonalize(X_csi, y);
    [testtt, testt, Th_vecIdx, ypred_vec] = diagonalize(X_csi, y_pred);
    [rsquare, rmse] = rsquare(y_vec,ypred_vec);
end