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
%   lambda, vector of lambdas to be tested
%   alpha,  vector of alphas to be tested
%   
%%% output:
% GCV, alphas*folds Matrix of GCV values obtained for each fold
% DF,  alphas*folds Matrix of estimated degrees of freedom for each fold
%

function [GCV,DF]=GCV_criterion_E_NET(OUTPUT,INPUT,folds,lambda,alpha)

train_perc=90; % 90% training 10% testing

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
    
    for kk=1:length(alpha)
        for hh=1:size(OUTPUT,2)
            [A(:,hh),stat]=lasso(yp_train',yn_train(:,hh),'Alpha',alpha(kk),'Lambda',lambda,'Standardize',1);
            MSE(hh)=stat.MSE;
        end
        DF(kk,it_f)=nnz(A);
%         MSE1(kk,it_f)=mean(MSE);
        RSS(kk,it_f) = mean(sum(abs(yn_test-yp_test'*A).^2, 1));
    end
    
    Nz=numel(A);
%     GCV(:,it_f)=MSE1(:,it_f)./((Nz-DF(:,it_f)).^2);
    GCV(:,it_f)=RSS(:,it_f)./((Nz-DF(:,it_f)).^2);
    
    
end
end
