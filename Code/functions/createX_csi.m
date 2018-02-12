function[X_csi] = createX_csi(X_L,X_R)
    Xcsi = cell(1,length(X_L));
    d_L = size(X_L{1},2);
    d_R = size(X_R,2);
    for i = 1:length(X_L)
       Xcsi{i} = zeros(size(X_L{i},1),d_L*d_R);
       ind = 1;
       for s = 1:d_L
           for t = 1:d_R
               tmp = X_L{i}(:,s).*X_R(i,t);
               Xcsi{i}(:,ind)=tmp;
               ind = ind + 1;
           end 
       end   
    end
    X_csi = Xcsi;
end

