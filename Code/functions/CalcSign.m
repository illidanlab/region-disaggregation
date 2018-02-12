function [accuracy,cm] = CalcSign(G,Gpred)
    if (size(G)~=size(Gpred))
        error('Estimate G and real G is not the same size');
    end
    lengthG = size(G,1)*size(G,2);
    G = reshape(G,[lengthG,1]);
    Gpred = reshape(Gpred,[lengthG,1]);
    Gposspa = double(G > 0);
    Gnegspa = -double(G < 0);
%     Gzerospa = double(G==0);
    Gspa = Gposspa + Gnegspa;
    Gpospred = double(Gpred > 0);
    Gnegpred = -double(Gpred < 0);
%     Gzeropred = double(Gpred==0);
    Gpredspa = Gpospred + Gnegpred;
%     Gpredspa = double(Gpred ~= 0);
    cm = confusionmat(Gpredspa',Gspa');
    accuracy = sum(diag(cm))/(sum(sum(cm,1)));
end