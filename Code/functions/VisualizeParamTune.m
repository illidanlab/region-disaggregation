function VisualizeParamTune(Lambda,perform_mat)
mcolor = mapminmax((1./perform_mat)',1,20);
% mcolor = mapminmax((perform_mat)',1,20);
figure;hold on;
for i = 1: length(Lambda)
    plot3(Lambda(i,1),Lambda(i,2),Lambda(i,3),'o','Markersize',mcolor(i),...
        'color',[1-mcolor(i)/20,1-mcolor(i)/20,1],'Markerfacecolor',[1-mcolor(i)/20,1-mcolor(i)/20,1]);hold on;
end
xlabel('\lambda_1');ylabel('\lambda_2');zlabel('\lambda3');grid on;

% outline the best
[~,best_idx] = min(perform_mat);
plot3(Lambda(best_idx,1),Lambda(best_idx,2),Lambda(best_idx,3),'o','Markersize',mcolor(best_idx),...
    'color',[1-mcolor(best_idx)/20,1-mcolor(best_idx)/20,1],'Markerfacecolor',[1-mcolor(best_idx)/20,1-mcolor(best_idx)/20,1],...
    'Markeredgecolor','r');

