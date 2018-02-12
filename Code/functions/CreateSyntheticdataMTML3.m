function [dataL,dataR,U,V,R] = CreateSyntheticdataMTML3()
% Generate synthetic data
% Output: data = year, lakeid, predictor, response
randn('state',2016);
rand('state',2016);
NumR = 20;
numset = round(abs(randn(1,NumR)*50)); % random generate # of lakes per region
X = [];y = [];
d = 10;
k = 5;
m = 4;
U = rand(d,m);
V = rand(m,NumR);
R = rand(k,m);

for i = 1:NumR
    xtmp = randn(numset(i),d)+i;
    ytmp = xtmp*U*V(:,i) + randn(size(xtmp,1),1);
    y = [y;ytmp];
    X =[X;[repmat(i,numset(i),1),xtmp]];
end
dataR = R*V+ randn(size(R*V));
dataR = dataR';
dataL = [(1:length(y))',X,y];



