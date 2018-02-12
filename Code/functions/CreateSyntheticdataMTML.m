function [dataL,Wtrue,dataR,Gtrue] = CreateSyntheticdataMTML2()
% Generate synthetic data
% Output: data = year, lakeid, predictor, response
randn('state',2016);
rand('state',2016);
NRegion = 20;
numset = round(abs(randn(1,NRegion)*20)); % random generate # of lakes per region
predictor = [];y1 = [];
Nfeature = 10;
k = 5;
Wtrue = zeros(Nfeature,NRegion);
Gtrue = randn(k,Nfeature); 
dataR = zeros(NRegion,k);
for i = 1:NRegion
    dataR(i,:) = randn(1,k);
    Wtrue(:,i) = Gtrue'*dataR(i,:)' + randn(Nfeature,1);
    xtmp = randn(numset(i),Nfeature)+i;
    ytmp = xtmp*Wtrue(:,i) + randn(size(xtmp,1),1);
    y1 = [y1;ytmp];
    predictor =[predictor;[repmat(i,numset(i),1),xtmp]];
end
dataL = [(1:length(y1))',predictor,y1];



