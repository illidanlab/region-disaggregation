function [ best_param, perform_mat] = TuneParam_MTML_Calibration_dropregion...
    ( obj_func_str, obj_func_opts, param_set, eval_func_str, higher_better,X,Y,vadratio,XR)
%     (X_train,y_train,X_csi_train, param_set,...
%     X_vali,y_vali,X_csi_vali,X_test,y_test,X_csi_test)
%     
%     numberofComb = size(param_set,1);
%     rsquare = zeros(numberofComb,1);
%     rmse_min = zeros(numberofComb,1);
%     
%     parfor i = 1: numberofComb
%         lambda1 = param_set(i,1);
%         lambda2 = param_set(i,2);
%         lambda3 = param_set(i,3);
%         [P,info,Th,q] = BCD_DFISTA(X_train, y_train, lambda1, lambda2, lambda3, X_csi_train);
%         [rsquaretmp,rmsetmp] = CalcError4Calibration(P,q,Xvali,yvali,X_csivali);
%         rsquare(i)=rsquaretmp;
%         rmse_min(i)=rmsetmp;
%     end 
%     [minrmse, bestInd] = min(rsquare);
%     lambda1 = param_set(bestInd,1);
%     lambda2 = param_set(bestInd,2);
%     lambda3 = param_set(bestInd,3);
%     [P_opt,info_opt,Th_opt,q_opt] = BCD_DFISTA(X_train, y_train, lambda1, lambda2, lambda3, X_csi_train);
%     [rsquare,rmse] = CalcError4Calibration(P_opt,q_opt,Xtest,ytest,X_csitest);
%      perform_mat = [rsquare,rmse];
%      best_param = [lambda1,lambda2,lambda3];
    eval_func = str2func(eval_func_str);
    obj_func  = str2func(obj_func_str);

    % performance vector
    perform_mat = zeros(size(param_set,1),1);

    [cv_Xtr, cv_Ytr, cv_Xte, cv_Yte] = splitTrnVal(X,Y,vadratio);
    for i = 1:length(cv_Xtr)
        cv_Xtr{i} = cv_Xtr{i}(:,2:end);
    end
    for i = 1:length(cv_Xte)
        cv_Xte{i} = cv_Xte{i}(:,2:end);
    end
    
%     cv_Xtr_LR = cell(length(cv_Xtr),1);
%     for i = 1:length(cv_Xtr)
%         cv_Xtr_LR{i} = horzcat(cv_Xtr{i},repmat(XR(i,:),size(cv_Xtr{i},1),1));
%     end
% 
%     cv_Xte_LR = cell(length(cv_Xte),1);
%     for i = 1:length(cv_Xtr)
%         cv_Xte_LR{i} = horzcat(cv_Xte{i},repmat(XR(i,:),size(cv_Xte{i},1),1));
%     end
    
    cv_Xcsitr = createX_csi(cv_Xtr,XR);
    cv_Xcsitst = createX_csi(cv_Xte,XR);
%     cv_Xtr_LR = reshape(cv_Xtr_LR,[1,86])
%     cv_Xte_LR = reshape(cv_Xte_LR,[1,86])
    delete(gcp('nocreate'))
    obj=parpool(24);
    parfor p_idx = 1: size(param_set,1)
        lambda1 = param_set(p_idx,1);
        lambda2 = param_set(p_idx,2);
        lambda3 = param_set(p_idx,3);
        [ P, info, Th, q ] = BCD_DFISTA(cv_Xtr, cv_Ytr, lambda1, lambda2, lambda3, cv_Xcsitr);
        perform_mat(p_idx) = eval_rmse_Calibration(cv_Xte,cv_Yte,cv_Xcsitst, P,q);
    end
    delete(obj);
    if(higher_better)
        [~,best_idx] = max(perform_mat);
    else
        [~,best_idx] = min(perform_mat);
    end
    best_param = param_set(best_idx,:);
end