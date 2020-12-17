%% Generalized cross-validation criterion
%  evaluation of GCV function as follows:
%
%                ||y - yp*A||^2
%          GCV= ----------------
%                 (N - df)^2
%
%  df = number of non zero for the matrix A estimated on the training set
%
%  ||y - yp*A||^2= Residual Sum of Squares
%
%%% input:
%   OUTPUT, N-p*M matrix of responses (y eq. 3)
%   INPUT,  Mp*(N-p) matrix of regressors (yp eq.3)
%   folds,  number of folds for cross-validation procedure 
%   lambda1, vector of lambdas1 to be tested
%   lambda2, vector of lambdas2 to be tested
%
%%% output:
% GCV, lambda1*folds matrix of GCV values obtained for each fold
% df,  lambda1*folds matrix of estimated degrees of freedom for each fold
%

function [GCV,DF]=GCV_criterion_F_LASSO(OUTPUT,INPUT,folds,lambda1,lambda2)

train_perc=90; % 90% training 10% testing
% parameters for fusedLeastR
opts.fusedPenalty=lambda2;
opts.rFlag=0; %for using lambda1 and lambda2 in the function
opts.rsL2=0; % for avoid elastic net approach
opts.init=2; % starting point to zero

for it_f=1:folds
    ind=randperm(size(OUTPUT,1));
    % dividing in train and testing set
    
    train_p=round(length(ind)*(train_perc/100));
    
    yp_train=INPUT(:,ind(1:train_p));
    yn_train=OUTPUT(ind(1:train_p),:);
    
    yp_test=INPUT(:,ind(train_p+1:end));
    yn_test=OUTPUT(ind(train_p+1:end),:);
    
    % standardize input and output matrices of training and testing
    yp_train=zscore(yp_train,0,2);
    yp_test=zscore(yp_test,0,2);
    yn_train=zscore(yn_train,0,1);
    yn_test=zscore(yn_test,0,1);
    
    for kk=1:length(lambda1)
        for hh=1:size(OUTPUT,2)
            [A(:,hh)]=fusedLeastR(yp_train', yn_train(:,hh), lambda1(kk), opts);
        end
        DF(kk,it_f)=nnz(A);
        RSS(kk,it_f) = mean(sum(abs(yn_test-yp_test'*A).^2, 1));
    end
    
    Nz=numel(A);
    GCV(:,it_f)=RSS(:,it_f)./((Nz-DF(:,it_f)).^2);
    
    
end
end
