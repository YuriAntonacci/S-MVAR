%% SPARSE IDENTIFICATION OF MVAR MODEL: Y(n)=A(1)Y(n-1)+...+A(p)Y(n-p)+U(n)
% makes use of Sparse Group-LASSO regression

%%% input:
% data, N*M matrix of time series (each time series is in a column)
% p, model order
% lambda1, lambdas1 x 1, vector of lambdas1 for optimal value selection
% lambda2, lambdas2 x 1, vector of lambdas2 for optimal values selection 
% folds, number of times for cross-validation approach

%%% output:
% lopt, estimated optimal lambda value
% GCVm,  lambda2*lambda1 GCV values averaged across folds
% df, lambda2*lambda1*folds matrix of estimated number of non-zero in A matrix
% Am=[A(1)...A(p)], M*pM matrix of the estimated MVAR model coefficients
% S, estimated M*M input covariance matrix

function [l1opt,l2opt,GCVm,df,Am,S] = SparseId_MVAR_SG_LASSO(data,p,lambda1,lambda2,folds)

y=data;
M=size(y,2);
N=size(y,1);
y=y';
INPUT=zeros(M*p,N-p);
OUTPUT=zeros(N-p,M);
%constructing predictors matrix
for m=1:M
    for i=1:N-p
        
        temp=[];
        for k=0:p-1,
            temp=[temp;y(:,i+k)];
        end
        INPUT(1:M*p,i)=temp;
        OUTPUT(i,m)=y(m,i+p);
    end
end

%%% Starting cross-validation
% Use the Parallel computing toolbox if exists
[~, ParallelToolBox]  = getWorkersAvailable();
if ParallelToolBox==1
    
    parfor ll=1:length(lambda2)
        [GCV(ll,:,:),df(ll,:,:)]=GCV_criterion_SG_LASSO(OUTPUT,INPUT,folds,lambda1,lambda2(ll));
    end
else
    for ll=1:length(lambda2)
        [GCV(ll,:,:),df(ll,:,:)]=GCV_criterion_SG_LASSO(OUTPUT,INPUT,folds,lambda1,lambda2(ll));
    end
    
end


% selecting optimal value of lambda
GCVm=mean(log10(GCV),3);
DEV=std(log10(GCV),0,3);
[MM,IND]=min(min(GCVm-DEV));
[I2,I1]=find((GCVm-DEV)==MM);
l1opt=lambda1(I1);
l2opt=lambda2(I2);
clear A y
% you can check at the optimal lambda by plotting GCVm (the lowest values)
% evaluate Am and SIGMA
% 
Y=data;
[l,m,t]=size(Y);
n=(l-p)*t;

%constructing predictors matrix
Yp=zeros(n,m,p);
for i=(p-1):-1:0
    Yp(:,:,p-i)=Y((i+1):end-(p-i),:);
end
Yp=permute(Yp,[1,3,2]);
Yp=reshape(Yp,[n,m*p]);
% constructing response vector
Yn=Y(p+1:end,:);

% parameters for SG-LASSO
ind=zeros(3,m);
IND=0:p:m*p;
ind(1,:)=IND(1:end-1)+1;
ind(2,:)=IND(2:end);
ind(3,:)=ones(1,m);


opts.rFlag=0; %0 for using lambda1 and lambda2 in the function 1 for values between [0 1]
opts.ind=ind;
opts.init=2; % 0 as starting point 
z=[l1opt,l2opt];
for i=1:size(Yn,2)
    [A(:,i)]=sgLeastR(Yp, Yn(:,i), z, opts);
end
A1=A';
Am=reshape(A1,[m,p,m]);
Am=permute(Am,[1,3,2]);

%predicting following samples
Yn_p=zeros(l,m,p);
for k=1:p
    bp=A1(:,k:p:end);
    Yn_p(p+1:end,:,k)=Y((p-(k-1)):(l-k),:)*bp';
end
Yn_p=sum(Yn_p,3);
Yn_p(1:p,:)=[];

%computing residuals
Up=Y(p+1:end,:)-Yn_p;
%computing residuals covariance matrix
S=cov(Up);

Am=reshape(Am,size(Am,1),[]);
end