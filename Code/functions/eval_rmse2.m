function [rmse,rmseall,r2,r2all,Y_pred,Y_real] = eval_rmse2 (X,Y,W)
% Nov.11 add rmseall,output rmse and r^2 for each task

task_num = length(X);
tmp = zeros(task_num,1);
rmseall = NaN(task_num,1);
r2all = NaN(task_num,1);
sample = zeros(task_num,1);
Y_pred = [];
Y_real = [];
for t = 1: task_num
    if(~isempty(Y{t}))
        y_pred = X{t} * W(:, t);
        tmp(t) = sum((y_pred - Y{t}).^2);
        sample(t) = length(y_pred);
        %     rmseall(t) = sqrt(tmp(t)/sample(t));
        [r2all(t),rmseall(t)] = rsquare(Y{t},y_pred);
            Y_pred = cat(1,Y_pred,y_pred);
            Y_real = cat(1,Y_real,Y{t});
    end
end
% mse = sum(tmp)/sum(sample);
%rmse = norm(Y_real-Y_pred)/length(Y_pred);
[r2,rmse] = rsquare(Y_real,Y_pred);
end
