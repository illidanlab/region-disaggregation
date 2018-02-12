function[Xtrain,Ytrain,Xvali,Yvali,Xlrtrn,Xlrvali,trnindex,tstindex] = splitTrnVal_3(X,Xlr,Y,vadratio)
    N = size(X,2);%num of task
    Xtrain = cell(1,N);
    Ytrain = cell(1,N);
    Xlrtrn = cell(1,N);
    Xlrvali = cell(1,N);
    Xvali = cell(1,N);
    Yvali = cell(1,N);
    trnindex = cell(1,N);
    tstindex = cell(1,N);
    Xtmp = X;
    Ytmp = Y;
    rng(2);
    for i = 1:N
        tmpsize = size(Xtmp{i},1);
        indperm = randperm(tmpsize);
        Xtmp{i} = Xtmp{i}(indperm,:);
        Ytmp{i} = Ytmp{i}(indperm,:);
        Xtrain{i} = Xtmp{i}(1:floor((1-vadratio)*tmpsize),:);
        Xlrtrn{i} = Xlr{i}(1:floor((1-vadratio)*tmpsize),:);
        Ytrain{i} = Ytmp{i}(1:floor((1-vadratio)*tmpsize),:);
        Xvali{i} = Xtmp{i}(floor((1-vadratio)*tmpsize)+1:end,:);
        Xlrvali{i} = Xlr{i}(floor((1-vadratio)*tmpsize)+1:end,:);
        Yvali{i} = Ytmp{i}(floor((1-vadratio)*tmpsize)+1:end,:);
        trnindex{i} = indperm(1:floor((1-vadratio)*tmpsize));
        tstindex{i} = indperm(floor((1-vadratio)*tmpsize)+1:end);
    end
end