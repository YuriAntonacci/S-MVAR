%% Conditional Granger Causality - theoretical example -
%%% evaluated with sparse identification of Multivariate VAR models
% analysis of simulated 5-variate VAR process 

clear; close all; clc;

load('TimeSeries.mat')

%%% MVAR process parameters

M=size(Am,1);
Su=eye(M);
p=size(Am,2)/M;
N=500; % Number of data samples (set the desired)
kratio=(N*M)/(M*M*p);
Y1=Y;
Y1=zscore(Y1(1:N*1.11,:),0,1);
Y=zscore(Y(1:N,:),0,1);

%% Theoretical conditional GC network

%%% ISS paramters
[A,C,K,V,Vy] = varma2iss(Am,[],Su,eye(M));

% % Conditional Granger Causality (eq. 13)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_js(jj,ii)=log(round(Sj_js,15)/round(Sj_ijs,15));
            
        end
    end
end
THEO=Fi_js;

%% conditional GC network - OLS -
disp('GC estimation - OLS')
tic
% MVAR model identification
[Am_OLS,Su_OLS,Yp_OLS,Up_OLS,Z_OLS,Yb_OLS]=idMVAR(Y',p,0);

%%% ISS paramters
[A,C,K,V,Vy] = varma2iss(Am_OLS,[],Su_OLS,eye(M));

% % Conditional Granger Causality (eq. 13)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_js(jj,ii)=log(round(Sj_js,15)/round(Sj_ijs,15));
            
        end
    end
end
cGC=Fi_js;

% testing significance for the estimated cGC values with surrogates 


[Fi_jsSurr]=cGCsurrogate(Y,100,p);
thr=prctile(Fi_jsSurr,95,3);
cGC(cGC<=thr)=0;
OLS=cGC;
toc

%% conditional GC network - LASSO -
tic
disp('GC estimation - LASSO')
%%% LASSO paramters
lambda=logspace(-1.5,0.2,200); % interval of lambdas
folds=10; %number of folds

% MVAR model identification
[lopt,GCV,df,Am_LASSO,Su_LASSO] = SparseId_MVAR_LASSO(Y1,p,lambda,folds);

%%% plot of GCV function
Fig1=figure;
set(Fig1(1),'Position',[196   469   560   418]);
plot(log10(lambda),GCV,'LineWidth',1.3)
xlabel('log( {\lambda} )');
ylabel('log (GCV)')
hold on
[ind]=find(lambda==lopt);
plot(log10(lopt),GCV(ind),'or','LineWidth',1.7)
tit=sprintf('Selected lambda = %s',num2str(lopt));
title(tit);

% ISS parameters
[A,C,K,V,Vy] = varma2iss(Am_LASSO,[],Su_LASSO,eye(M)); %

% % Conditional Granger Causality (eq. 13)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_js(jj,ii)=log(round(Sj_js,15)/round(Sj_ijs,15));

        end
    end
end
LASSO=Fi_js;
toc
%%  conditional GC network - Elastic-NET
tic
disp('GC estimation - Elastic Net')
%%% EL-NET paramters
lambda=logspace(-1,1,100); % interval of lambdas
alpha=[0:0.05:0.9];
alpha(1)=0.01;%intervals of alphas
folds=10; %number of folds

% MVAR model identification
[lopt,aopt,GCV,df,Am_ENET,Su_ENET] = SparseId_MVAR_E_NET(Y1,p,lambda,alpha,folds);

%%% plot of GCV function
figure2=figure;
axes1 = axes('Parent',figure2);
colormap hsv
surf(log10(lambda'),log10(alpha'),GCV','FaceColor','interp','EdgeColor','none','FaceLighting','gouraud');
axis tight
view(axes1,[134 14]);
camlight left
ylabel('log alpha','FontName','TimesNewRoman','FontSize',12,'FontWeight','Bold');
xlabel('log lambda','FontName','TimesNewRoman','FontSize',12,'FontWeight','Bold');
zlabel('GCV','FontName','TimesNewRoman','FontSize',12,'FontWeight','Bold');
            
%%% ISS parameters
[A,C,K,V,Vy] = varma2iss(Am_ENET,[],Su_ENET,eye(M)); %


% % Conditional Granger Causality (eq. 13)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_js(jj,ii)=log(round(Sj_js,4)/round(Sj_ijs,4));

        end
    end
end
E_NET=Fi_js;
toc

%% conditional GC network - Fused LASSO regression
tic
disp('GC estimation - Fused LASSO')
%%% F-LASSO paramters
lambda1=logspace(1,3,80); % interval of lambdas
lambda2=logspace(1,3,40);
folds=10; %number of folds

% MVAR model identification
[l1opt,l2opt,GCV,df,Am_FLASSO,Su_FLASSO] = SparseId_MVAR_F_LASSO(Y1,p,lambda1,lambda2,folds);

%%% plot of GCV function
figure3=figure;
axes1 = axes('Parent',figure3);
colormap hsv
surf(log10(lambda2'),log10(lambda1'),GCV','FaceColor','interp','EdgeColor','none','FaceLighting','gouraud');
axis tight
view(axes1,[134 14]);
camlight left
ylabel('log lambda1','FontName','TimesNewRoman','FontSize',12,'FontWeight','Bold');
xlabel('log lambda2','FontName','TimesNewRoman','FontSize',12,'FontWeight','Bold');
zlabel('GCV','FontName','TimesNewRoman','FontSize',12,'FontWeight','Bold');
            

%%% ISS parameters
[A,C,K,V,Vy] = varma2iss(Am_FLASSO,[],Su_FLASSO,eye(M)); %

% % Conditional Granger Causality (eq. 13)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_js(jj,ii)=log(round(Sj_js,4)/round(Sj_ijs,4));

        end
    end
end
F_LASSO=Fi_js;
toc

%% conditional GC network - Sparse Group LASSO regression
tic
disp('GC estimation - Sparse Group LASSO')
%%% SG-LASSO paramters
lambda1=logspace(1,3,40); % interval of lambdas
lambda2=logspace(1,3,20);
folds=10; %number of folds

%%% MVAR model identification
[l1opt,l2opt,GCV,df,Am_SGLASSO,Su_SGLASSO] = SparseId_MVAR_SG_LASSO(Y1,p,lambda1,lambda2,folds);

%%% plot of GCV function
figure4=figure;
axes1 = axes('Parent',figure4);
colormap hsv
surf(log10(lambda2'),log10(lambda1'),GCV','FaceColor','interp','EdgeColor','none','FaceLighting','gouraud');
axis tight
view(axes1,[134 14]);
camlight left
ylabel('log lambda1','FontName','TimesNewRoman','FontSize',12,'FontWeight','Bold');
xlabel('log lambda2','FontName','TimesNewRoman','FontSize',12,'FontWeight','Bold');
zlabel('GCV','FontName','TimesNewRoman','FontSize',12,'FontWeight','Bold');
            
%%% ISS parameters
[A,C,K,V,Vy] = varma2iss(Am_SGLASSO,[],Su_SGLASSO,eye(M)); %


% % Conditional Granger Causality (eq. 13)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_js(jj,ii)=log(round(Sj_js,4)/round(Sj_ijs,4));

        end
    end
end
SG_LASSO=Fi_js;
toc

%% plot of cGC networks

figure
subplot(1,6,1);
plot_pw(THEO);
title('Theo');

subplot(1,6,2);
plot_pw(OLS);
title('OLS');

subplot(1,6,3);
plot_pw(LASSO);
title('LASSO')

subplot(1,6,4);
plot_pw(E_NET);
title('E-NET')

subplot(1,6,5);
plot_pw(F_LASSO);
title('F-LASSO')

subplot(1,6,6);
plot_pw(SG_LASSO);
title('SG-LASSO')


tit=sprintf('Samples=%s, Kratio=%s',num2str(N),num2str(kratio));
suptitle(tit)

