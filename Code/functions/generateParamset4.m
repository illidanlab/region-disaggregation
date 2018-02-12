function [Lambda] = generateParamset4(lmd1,lmd2,lmd3,lmd4)
Lambda = zeros(length(lmd1)*length(lmd2)*length(lmd3)*length(lmd4),4);
cnt = 1;
for l1 = 1:length(lmd1)
    for l2 = 1:length(lmd2)
        for l3 = 1:length(lmd3)
            for l4 = 1:length(lmd4)
                Lambda(cnt,:) = [lmd1(l1),lmd2(l2),lmd3(l3),lmd4(l4)];
                cnt = cnt+1; 
            end
        end
    end
end

    