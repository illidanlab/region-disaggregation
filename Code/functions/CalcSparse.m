function [cm, accuracy, precision, recall, F1] = CalcSparse(G,Gpred)
    if (size(G)~=size(Gpred))
        error('Estimate G and real G is not the same size');
    end
    lengthG = size(G,1)*size(G,2);
    G = reshape(G,[lengthG,1]);
    Gpred = reshape(Gpred,[lengthG,1]);
    Gspa = double(G ~= 0);
    Gpredspa = double(Gpred ~= 0);
    [~,cm,~,~] = confusion(Gspa',Gpredspa');
    tp = cm(1,1);
    tn = cm(2,2);
    fp = cm(1,2);
    fn = cm(2,1);
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    accuracy = (tp+tn)/(tp+tn+fp+fn);
    F1 = 2*precision*recall/(precision+recall);
    if isnan(F1) == 1
       F1 = 0;
    end
end