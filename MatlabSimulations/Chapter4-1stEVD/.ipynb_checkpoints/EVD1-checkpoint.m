% This code is written to support the experiments in the book titled:
% Adaptive Machine Learning Algorithms with Python
% by
% Chanchal Chatterjee
% December 2021
% Comparison of Adaptive First Principal Compt EVD

function evdTest;

clear all; 
close all; 

% Number of samples N and dimension D
nSamples = 500;
nDim     = 10;
nEpochs  = 2;
randn('state',0);

% Generate the covX random matrices
covX = 3*...
[ 0.4270   0.0110  -0.0050  -0.0250   0.0890  -0.0790  -0.0190   0.0740   0.0890   0.0050 ; ...
  0.0110   5.6900  -0.0690  -0.2820  -0.7310   0.0900  -0.1240   0.1000   0.4320  -0.1030 ; ...
 -0.0050  -0.0690   0.0800   0.0980   0.0450  -0.0410   0.0230   0.0220  -0.0350   0.0120 ; ...
 -0.0250  -0.2820   0.0980   2.8000  -0.1070   0.1500  -0.1930   0.0950  -0.2260   0.0460 ; ...
  0.0890  -0.7310   0.0450  -0.1070   3.4400   0.2530   0.2510   0.3160   0.0390  -0.0100 ; ...
 -0.0790   0.0900  -0.0410   0.1500   0.2530   2.2700  -0.1800   0.2950  -0.0390  -0.1130 ; ...
 -0.0190  -0.1240   0.0230  -0.1930   0.2510  -0.1800   0.3270   0.0270   0.0260  -0.0160 ; ...
  0.0740   0.1000   0.0220   0.0950   0.3160   0.2950   0.0270   0.7270  -0.0960  -0.0170 ; ...
  0.0890   0.4320  -0.0350  -0.2260   0.0390  -0.0390   0.0260  -0.0960   0.7150  -0.0090 ; ...
  0.0050  -0.1030   0.0120   0.0460  -0.0100  -0.1130  -0.0160  -0.0170  -0.0090   0.0650 ];
    
% Compute data
[V1,D1] = eig(covX);
[d,in] = sort(diag(D1));
V = zeros(nDim,nDim);
D = zeros(nDim,nDim);
for i = 1 : nDim
   V(:,i) = V1(:,in(nDim+1-i));
   D(i,i) = D1(in(nDim+1-i),in(nDim+1-i));
end
x(1:nDim,1:nSamples) = V*sqrt(D)*randn(nDim,nSamples);
ActualD = diag(D)'

% Compute Corrl matrix and eigen vectors
corX = zeros(nDim,nDim);
for i = 1 : nSamples
   corX = corX + x(:,i)*x(:,i)'/nSamples;
end
[V1,D1] = eig(corX);
[d,in] = sort(diag(D1));
V = zeros(nDim,nDim);
D = zeros(nDim,nDim);
for i = 1 : nDim
   V(:,i) = V1(:,in(nDim+1-i));
   D(i,i) = D1(in(nDim+1-i),in(nDim+1-i));
end
corX;
EstV = V(:,1)'
EstD = diag(D)'

% Adaptive algorithm
A = zeros(nDim,nDim);
w = ones(nDim, 11);
for i = 1 : 11
    w(:,i) = w(:,i) / norm(w(:,i));
end
w0 = w;
mu = 10;
%w(:,11) = 0.1*ones(nDim, 1);

for epoch = 1 : nEpochs
    for iter = 1 : nSamples
        
        cnt = nSamples*(epoch-1) + iter;
      
        A = A + (1.0/cnt)*((x(:,iter) * x(:,iter)') - A);
        etat = 1.0/(150 + cnt);
      
        % OJA
        w(:,1) = w(:,1) + etat*(A*w(:,1) - w(:,1)*(w(:,1)'*A*w(:,1)));
      
        % OJAN
        w(:,2) = w(:,2) + etat*( A*w(:,2) - w(:,2)* ((w(:,2)'*A*w(:,2)) / (w(:,2)'*w(:,2))) );
      
        % LUO
        w(:,3) = w(:,3) + etat*(w(:,3)'*w(:,3))*( A*w(:,3) - w(:,3)* ((w(:,3)'*A*w(:,3)) / (w(:,3)'*w(:,3))) );
      
        % RQ
        w(:,4) = w(:,4) + etat*(1.0/(w(:,4)'*w(:,4)))*(A*w(:,4) - w(:,4)*((w(:,4)'*A*w(:,4)) / (w(:,4)'*w(:,4))));
      
        % OJA+
        w(:,5) = w(:,5) + etat*(A*w(:,5) - w(:,5)*(w(:,5)'*A*w(:,5)) - w(:,5)*(1 - w(:,5)'*w(:,5)));
      
        % IT
        w(:,6) = w(:,6) + etat*((A*w(:,6) / (w(:,6)'*A*w(:,6))) - w(:,6));
      
        % XU
        w(:,7) = w(:,7) + etat*(2*A*w(:,7) - w(:,7)*(w(:,7)'*A*w(:,7)) - A*w(:,7)*(w(:,7)'*w(:,7)));
      
        % PF
        w(:,8) = w(:,8) + etat*(A*w(:,8) - mu*w(:,8)*(w(:,8)'*w(:,8) - 1));
      
        % AL1
        w(:,9) = w(:,9) + etat*(A*w(:,9) - w(:,9)*(w(:,9)'*A*w(:,9)) - mu*w(:,9)*(w(:,9)'*w(:,9) - 1));
      
        % AL2
        w(:,10) = w(:,10) + etat*(2*A*w(:,10) - w(:,10)*(w(:,10)'*A*w(:,10)) - A*w(:,10)*(w(:,10)'*w(:,10)) - mu*w(:,9)*(w(:,10)'*w(:,10) - 1));
      
        % FENG
        w(:,11) = w(:,11) + (1.0/(17000+cnt))*((w(:,11)'*w(:,11))*A*w(:,11) - w(:,11));
      
        for i = 1 : 11
            u(:,i) = w(:,i) / norm(w(:,i));
      	    cos_t(i,cnt) = abs(u(:,i)'*V(:,1));
            ind(cnt) = cnt;
		end
	end
end


for i = 1 : 11
	ww(:,i) = (w(:,i)'*w(:,i));
end
ww

for i = 1 : 11
	EstL(i) = (w(:,i)'*A*w(:,i)) / (w(:,i)'*w(:,i));
end
EstL


if (1)

% Plot the convergence results
figure(1);
plot(ind, cos_t(1,:),'k-'); hold on;
plot(ind, cos_t(2,:),'k:'); hold on;
legend('OJA','OJAN',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

% Plot the convergence results
figure(2);
plot(ind, cos_t(3,:),'k-'); hold on;
plot(ind, cos_t(4,:),'k:'); hold on;
legend('LUO','RQ',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

% Plot the convergence results
figure(3);
plot(ind, cos_t(5,:),'k-'); hold on;
plot(ind, cos_t(6,:),'k:'); hold on;
legend('OJA+','IT',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

% Plot the convergence results
figure(4);
plot(ind, cos_t(7,:),'k-'); hold on;
plot(ind, cos_t(8,:),'k:'); hold on;
legend('XU','PF',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

% Plot the convergence results
figure(5);
plot(ind, cos_t(9,:),'k-'); hold on;
plot(ind, cos_t(10,:),'k:'); hold on;
plot(ind, cos_t(11,:),'k-.'); hold on;
legend('AL1','AL2','FENG',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

else

% Plot the convergence results
figure(1);
plot(ind,cos_t(1,:),'r:'); hold on;
plot(ind,cos_t(2,:),'g-'); hold on;
plot(ind,cos_t(3,:),'b--'); hold on;
legend('OJA','OJAN','LUO',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

% Plot the convergence results
figure(2);
plot(ind,cos_t(4,:),'r:'); hold on;
plot(ind,cos_t(5,:),'g-'); hold on;
plot(ind,cos_t(6,:),'b--'); hold on;
legend('RQ','OJA+','IT',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

% Plot the convergence results
figure(3);
plot(ind,cos_t(7,:),'r:'); hold on;
plot(ind,cos_t(8,:),'g-'); hold on;
plot(ind,cos_t(9,:),'b--'); hold on;
legend('XU','PF','AL1',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

% Plot the convergence results
figure(4);
plot(ind,cos_t(9,:),'r:'); hold on;
plot(ind,cos_t(10,:),'g-'); hold on;
plot(ind,cos_t(5,:),'b--'); hold on;
legend('AL1','AL2','OJA+',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

% Plot the convergence results
figure(5);
plot(ind,cos_t(11,:),'r:'); hold on;
plot(ind,cos_t(12,:),'g-'); hold on;
legend('AL1','FENG',0);
xlabel('Number of Samples');
ylabel('Direction Cosines');
hold off;

end

