function rmse = eval_rmse (X,Y,W)
task_num = length(X);
tmp = zeros(task_num,1);
sample = zeros(task_num,1);

for t = 1: task_num
    y_pred = X{t} * W(:, t);
    tmp(t) = sum((y_pred - Y{t}).^2);
    sample(t) = length(y_pred);
end
mse = sum(tmp)/sum(sample);
rmse = sqrt(mse);
end
