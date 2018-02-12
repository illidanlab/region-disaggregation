function [ best_idx, perform_mat] = TuneParam_Xlr...
    (param_set, X, Y,X_R,coord,trainIdx,testIdx,vadratio)
%% INPUT
% data: original data
%   obj_func_str:  1-parameter optimization algorithms
%   param_set:   the set of the parameter. array
%   eval_func_str: evaluation function:
%       signature [performance_measure] = eval_func(Y_test, X_test, W_learnt)
%   higher_better: if the performance is better given
%           higher measurement (e.g., Accuracy, AUC)
%% OUTPUT
%   best_param:  best parameter in the given parameter range
%   perform_mat: the average performance for every parameter in the
%                parameter range.
% eval_func = str2func(eval_func_str);
% obj_func  = str2func(obj_func_str);

% performance vector
perform_mat = zeros(size(param_set,1),5);
r = length(Y);
cvind = cell(1,length(Y));
for i = 1:length(Y)
   cvind{i} =  crossvalind('kfold',size(Y{i},1),5);
end


teIdx = cell(1,r);
trIdx = cell(1,r);
cv_Xtr = cell(1,r);
cv_Xte = cell(1,r);
cv_Ytr = cell(1,r);
cv_Yte = cell(1,r);
for inner = 1:5
    for i = 1:r
        teIdx{i} = find(cvind{i}==inner);
        trIdx{i} = find(cvind{i}~=inner);
        cv_Xtr{i} = X{i}(trIdx{i},:);
        cv_Xte{i} = X{i}(teIdx{i},:);
        cv_Ytr{i} = Y{i}(trIdx{i},:);
        cv_Yte{i} = Y{i}(teIdx{i},:);
    end
    delete(gcp('nocreate'));
%     obj = parpool(2);
    for p_idx = 1: size(perform_mat,1)
        [Wl,Wr,Gestimate,~,Xlrte,~] = MTML_imputation(cv_Ytr,cv_Xtr,cv_Xte,X_R,...,
            param_set(p_idx,1),param_set(p_idx,2),param_set(p_idx,3),param_set(p_idx,4),coord,randn(size(cv_Xtr{1},2),size(X_R,2)),trIdx,teIdx,50,1e-5);
        ypredcv = cell(1,r);
        xall = cell(1,r);
    %     GestimateBi = double(Gestimate~=0);
    %     gammabi = reshape(GestimateBi',[dl*dr,1]);
    %     indnzero = find(gammabi~=0);
    %     Xlrcsite = createX_csi_local(cv_Xte,Xlrte);

    %     for i = 1:r
    %         Xlrcsite{i} = Xlrcsite{i}(:,indnzero);
    %     end

        for i = 1:r
            xall{i} = horzcat(cv_Xte{i},Xlrte{i});
        end
        ytmp = cv_Yte;

        for i = 1:r
            ytmp{i} = cv_Yte{i} - diag(cv_Xte{i}*Gestimate*Xlrte{i}');
        end
         b = Least_Lasso(xall,ytmp,0);
        for i = 1:r
    %         ypredcv{i} = cv_Xte{i}*Wl(:,i) +  Xlrte{i} * Wr(:,i) + diag(cv_Xte{i}*Gestimate*Xlrte{i}');
              ypredcv{i} = xall{i}*b(:,i) + diag(cv_Xte{i}*Gestimate*Xlrte{i}');
        end
        Ytstcvall = cat(1,cv_Yte{:});
        Ypredtstcvall =cat(1,ypredcv{:});
        perform_mat(p_idx,inner) = norm(Ytstcvall-Ypredtstcvall)/length(Ytstcvall);
    end
% delete(obj);
end
% [cv_Xtr, cv_Ytr, cv_Xte, cv_Yte,trIdx,teIdx] = splitTrnVal_ind(X,Y,vadratio);
% dl = size(cv_Xtr{1},2);
% dr = size(X_R,2);
% % delete(gcp('nocreate'));
% % obj = parpool(24);
% for p_idx = 1: length(perform_mat)
%     [Wl,Wr,Gestimate,~,Xlrte,~] = MTML_imputation(cv_Ytr,cv_Xtr,cv_Xte,X_R,...,
%         param_set(p_idx,1),param_set(p_idx,2),param_set(p_idx,3),param_set(p_idx,4),coord,randn(size(cv_Xtr{1},2),size(X_R,2)),trIdx,teIdx,50,1e-5);
%     ypredcv = cell(1,r);
%     xall = cell(1,r);
% %     GestimateBi = double(Gestimate~=0);
% %     gammabi = reshape(GestimateBi',[dl*dr,1]);
% %     indnzero = find(gammabi~=0);
% %     Xlrcsite = createX_csi_local(cv_Xte,Xlrte);
%     
% %     for i = 1:r
% %         Xlrcsite{i} = Xlrcsite{i}(:,indnzero);
% %     end
%     
%     for i = 1:r
%         xall{i} = horzcat(cv_Xte{i},Xlrte{i});
%     end
% %     ytmp = cv_Yte;
%     
%     for i = 1:r
%         ytmp{i} = cv_Yte{i} - diag(cv_Xte{i}*Gestimate*Xlrte{i}');
%     end
%      b = Least_L21(xall,ytmp,0.01);
%     for i = 1:r
% %         ypredcv{i} = cv_Xte{i}*Wl(:,i) +  Xlrte{i} * Wr(:,i) + diag(cv_Xte{i}*Gestimate*Xlrte{i}');
%           ypredcv{i} = xall{i}*b(:,i) + diag(cv_Xte{i}*Gestimate*Xlrte{i}');
%     end
%     Ytstcvall = cat(1,cv_Yte{:});
%     Ypredtstcvall =cat(1,ypredcv{:});
%     perform_mat(p_idx) = norm(Ytstcvall-Ypredtstcvall)/length(Ytstcvall);
% end
% delete(obj);
perform_mat = mean(perform_mat,2);
[~,best_idx] = min(perform_mat);
end

