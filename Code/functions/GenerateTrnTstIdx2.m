function [Trnidx,Tstidx,trnnum,tstnum] = GenerateTrnTstIdx2(data, trnrate,regionratio,randnumber)
% Input:
% data: 1st col = lakeid, 2nd col = eduid, 3rd col = response var, 4 to end
% = features.
% splitratio = trnrate.

% Feb.5
% select regions that corresponding lakes are not in the training set
if nargin<3
    randn('state',2016);
    rand('state',2016);
else
    randn('state',2016+randnumber);
    rand('state',2016+randnumber);
end

Eduid = unique(data(:,2));
NumEdu = length(Eduid); % number of tasks
Trnidx = [];
Tstidx = [];
trnnum = zeros(NumEdu,1);
tstnum = zeros(NumEdu,1);
Includeidx1 = find(~isnan(data(:,3)));

include_region = Eduid(randperm(length(Eduid),round(length(Eduid)*(1-regionratio))));
% Includeidx2 = find(ismember(data(:,2),include_region));
% Includeidx = intersect(Includeidx1,Includeidx2);
for id = 1: NumEdu
    eduid = Eduid(id);    
    idx = find(data(:,2)==eduid);
    idx = idx(ismember(idx,Includeidx1));
    if ismember(eduid,include_region)
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
    else
        Tstidx = [Tstidx;idx];
        tstnum(id) = length(idx);
        
    end
    
end