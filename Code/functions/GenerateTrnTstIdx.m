function [Trnidx,Tstidx,trnnum,tstnum] = GenerateTrnTstIdx(data, trnrate,randnumber)
% Input:
% data: 1st col = Year, 2nd col = lakeid, 3rd col = response var, 4 to end
% = features.
% splitratio = trnrate.
if nargin<3
    randn('state',2016);
    rand('state',2016);
else
    randn('state',2016+randnumber);
    rand('state',2016+randnumber);
end

Lakeid = unique(data(:,2));
NumLake = length(Lakeid); % number of tasks
Trnidx = [];
Tstidx = [];
trnnum = zeros(NumLake,1);
tstnum = zeros(NumLake,1);
Includeidx = find(~isnan(data(:,3)));
for id = 1: NumLake
    lakeid = Lakeid(id);
    idx = find(data(:,2)==lakeid);
    idx = idx(ismember(idx,Includeidx));
    num_years = length(idx);
    if num_years == 0
        continue;
    elseif num_years ==1
        Tstidx = [Tstidx;idx];
%         trnnum(id) = 0;
        tstnum(id) = 1;
    else
        randpermidx = randperm(num_years);
        trnidx = idx(randpermidx(1:round(num_years*trnrate)));
        tstidx = setdiff(idx,trnidx);
        Trnidx = [Trnidx;trnidx];
        Tstidx = [Tstidx;tstidx];
        trnnum(id) = length(trnidx);
        tstnum(id) = length(tstidx);
    end
end