function [SSB,SSW] = ComputeSSBSSW(DataAfterPca, ClassIdx )
%Metric
%1 - ssw
%2 - ssb
%Classidx matrix of N by 1
SSB = zeros(size(ClassIdx,2),1);
SSW = zeros(size(ClassIdx,2),1);
unique_classidx = unique(ClassIdx);
for j = 1 : size(ClassIdx,2)
    classidx = ClassIdx(:,j);
    num_cluster = length(unique_classidx);
    ssw = zeros(num_cluster,1);
    ssb = zeros(num_cluster,1);
    for i = 1: num_cluster
        ci = mean(DataAfterPca(classidx ==unique_classidx(i),:),1);
        ssw(i) = sum(sum((DataAfterPca(classidx ==unique_classidx(i),:)-...
            repmat(ci,sum(classidx==unique_classidx(i)),1)).^2));
        
        c = mean(DataAfterPca,1);
        ssb(i) = sum(classidx ==unique_classidx(i))*sum((ci - c).^2);
    end
    SSW(j,1) = sum(ssw);
    SSB(j,1) = sum(ssb);
    
end
end
