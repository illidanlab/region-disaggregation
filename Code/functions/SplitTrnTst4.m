function [Xtrn, Ytrn, Xtst, Ytst] = SplitTrnTst4(data, Trnidx,Tstidx,coord)
% Input:
% data: 1st col = Year, 2nd col = lakeid, 3rd col = response var, 4 to end
% = features.
% Trnidx Tstidx is specified. 
% SplitTrnTst3 - is not specified.
% Dec.9
% Split randomly, priority: tst, trn, vad.

Lakeid = unique(data(:,2));
NumLake = length(Lakeid); % number of tasks
Xtrn = cell(1,NumLake);
Ytrn = cell(1,NumLake);
Xtst = cell(1,NumLake);
Ytst = cell(1,NumLake);
coordtrn = cell(1,NumLake);
coordtst = cell(1,Numlake);
for id = 1: NumLake
    lakeid = Lakeid(id);
    idx = find(data(:,2)==lakeid);    
    trnidx = idx(ismember(idx,Trnidx));
    tstidx = idx(ismember(idx,Tstidx));
    Xtrn{1,id} = data(trnidx,4:end);
    Xtst{1,id} = data(tstidx,4:end);
    Ytrn{1,id} = data(trnidx,3);
    Ytst{1,id} = data(tstidx,3);
    coordtrn{1,id} = coord(trnidx,:);
    coordtst{1,id} = coord(tstidx,:);
end
