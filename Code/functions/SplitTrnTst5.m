function [Xtrn, Ytrn, Xtst, Ytst,NumLake,trnnum,tstnum,Trnidx,Tstidx] = SplitTrnTst5(data,Lakeid,trnrate)
% Input:
% data: 1st col = Year, 2nd col = lakeid, 3rd col = response var, 4 to end
% = features.
% splitratio = trnrate.

% Dec.9
% Split randomly, priority: tst, trn, vad.
% Dec.28 
% Add id as input to avoid incosistency with high level features;
randn('state',2016);
rand('state',2016);

% Lakeid = unique(data(:,2));
NumLake = length(Lakeid); % number of tasks
Xtrn = cell(1,NumLake);
Ytrn = cell(1,NumLake);
Xtst = cell(1,NumLake);
Ytst = cell(1,NumLake);
trnnum = zeros(NumLake,1);
tstnum = zeros(NumLake,1);
Trnidx = [];
Tstidx = [];
if any(any(isnan(data)))
    Includeidx = find(~isnan(data(:,3)));
    for id = 1: NumLake
        lakeid = Lakeid(id);
        idx = find(data(:,2)==lakeid);
        idx = idx(ismember(idx,Includeidx));
        num_years = length(idx);
        if num_years ==0
            trnnum(id) = 0;
            Xtrn{1,id} = data([],4:end);
            Ytrn{1,id} = data([],3);
            Xtst{1,id} = data([] ,4:end);
            Ytst{1,id} = data([],3);
            tstnum(id) = 0;  
        elseif num_years ==1
            trnnum(id) = 0;
            Xtrn{1,id} = data([],4:end);
            Ytrn{1,id} = data([],3);
            Xtst{1,id} = data(idx ,4:end);
            Ytst{1,id} = data(idx,3);
            tstnum(id) = 1;
            Tstidx = cat(1,Tstidx,idx);
        else
            randpermidx = randperm(num_years);
            trnidx = idx(randpermidx(1:round(num_years*trnrate)));
            tstidx = setdiff(idx,trnidx);
            Trnidx = cat(1,Trnidx,trnidx);
            Tstidx = cat(1,Tstidx,tstidx);
            Xtrn{1,id} = data(trnidx,4:end);
            Xtst{1,id} = data(tstidx,4:end);
            Ytrn{1,id} = data(trnidx,3);
            Ytst{1,id} = data(tstidx,3);
            trnnum(id) = length(trnidx);
            tstnum(id) = length(tstidx);
        end
    end
else
    for id = 1: NumLake
        lakeid = Lakeid(id);
        idx = find(data(:,2)==lakeid);
        num_years = length(idx);
        if num_years ==1
            trnnum(id) = 0;
            Xtrn{1,id} = data([],4:end);
            Ytrn{1,id} = data([],3);
            Xtst{1,id} = data(idx ,4:end);
            Ytst{1,id} = data(idx,3);
            tstnum(id) = 1;
            Tstidx = [Tstidx;idx];
        else
            randpermidx = randperm(num_years);
            trnidx = idx(randpermidx(1:round(num_years*trnrate)));
            tstidx = setdiff(idx,trnidx);
            Trnidx = [Trnidx;trnidx];
            Tstidx = [Tstidx;tstidx];
            Xtrn{1,id} = data(trnidx,4:end);
            Xtst{1,id} = data(tstidx,4:end);
            Ytrn{1,id} = data(trnidx,3);
            Ytst{1,id} = data(tstidx,3);
            trnnum(id) = length(trnidx);
            tstnum(id) = length(tstidx);
        end
    end
end

% fprintf('%i out of %i tasks has training data, with total %i training instances. \n',NumLake - sum(cellfun(@isempty,Xtrn)),NumLake, sum(trnnum));
% fprintf('%i out of %i tasks has test data, with total %i test instances. \n',NumLake - sum(cellfun(@isempty,Xtst)),NumLake,sum(tstnum));

