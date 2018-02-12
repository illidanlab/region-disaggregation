function [Lambda] = generateParamset2(lmd1,lmd2)
Lambda = zeros(length(lmd1)*length(lmd2),2);
cnt = 1;
for l1 = 1:length(lmd1)
    for l2 = 1:length(lmd2)
        
            Lambda(cnt,:) = [lmd1(l1),lmd2(l2)];
            cnt = cnt+1;
        
    end
end

    