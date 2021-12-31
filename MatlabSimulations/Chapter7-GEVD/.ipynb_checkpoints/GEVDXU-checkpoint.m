% This code is written to support the experiments in the book titled:
% Adaptive Machine Learning Algorithms with Python
% by
% Chanchal Chatterjee
% December 2021
% XU - Comparison of Adaptive GEVD Homogeneous, Deflated, Weighted

function GEVDXU;

clear all; 
close all; 

% Number of samples N and dimension D
nSamples = 500;
nDim     = 10;
nEA      = 4;
nEpochs  = 3;
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

covY = 2*...
[ 0.3350   0.0260  -0.0510  -0.0120   0.0790   0.0170   0.0290   0.0080   0.0770  -0.0300 ; ...
  0.0260   0.0910   0.0110  -0.0100   0.0060  -0.0140  -0.0020  -0.0230   0.0110   0.0350 ; ...
 -0.0510   0.0110   0.0780   0.0000   0.0160   0.0030   0.0300  -0.0350  -0.0030  -0.0490 ; ...
 -0.0120  -0.0100   0.0000   0.0820  -0.0030  -0.0260  -0.0250  -0.0290  -0.0150   0.0250 ; ...
  0.0790   0.0060   0.0160  -0.0030   0.7970   0.1940  -0.0370  -0.0230   0.0590  -0.1450 ; ...
  0.0170  -0.0140   0.0030  -0.0260   0.1940   1.5000   0.0140  -0.1040   0.1140  -0.2290 ; ...
  0.0290  -0.0020   0.0300  -0.0250  -0.0370   0.0140   0.2770  -0.0300  -0.0770  -0.0510 ; ...
  0.0080  -0.0230  -0.0350  -0.0290  -0.0230  -0.1040  -0.0300   0.3170   0.0220   0.0100 ; ...
  0.0770   0.0110  -0.0030  -0.0150   0.0590   0.1140  -0.0770   0.0220   0.5380   0.0340 ; ...
 -0.0300   0.0350  -0.0490   0.0250  -0.1450  -0.2290  -0.0510   0.0100   0.0340   0.6680 ];

%covX
%covY
%[Vxx,Dxx] = eig(covX, covY);
%sort(Dxx)

% Compute X samples
[V1,D1] = eig(covX);
[d,in] = sort(diag(D1));
V = zeros(nDim,nDim);
D = zeros(nDim,nDim);
for i = 1 : nDim
    V(:,i) = V1(:,in(nDim+1-i));
    D(i,i) = D1(in(nDim+1-i),in(nDim+1-i));
end
x(1:nDim,1:nSamples) = V*sqrt(D)*randn(nDim,nSamples);

% Compute Y samples
[V1,D1] = eig(covY);
[d,in] = sort(diag(D1));
V = zeros(nDim,nDim);
D = zeros(nDim,nDim);
for i = 1 : nDim
    V(:,i) = V1(:,in(nDim+1-i));
    D(i,i) = D1(in(nDim+1-i),in(nDim+1-i));
end
y(1:nDim,1:nSamples) = V*sqrt(D)*randn(nDim,nSamples);

% Compute Corrl matrix and eigen values for X samples
corX = zeros(nDim,nDim);
for i = 1 : nSamples
    corX = corX + (x(:,i) * x(:,i)' / nSamples);
end
[V1,D1] = eig(corX);
[d,in] = sort(diag(D1));
DX = zeros(nDim,nDim);
for i = 1 : nDim
    DX(i,i) = D1(in(nDim+1-i),in(nDim+1-i));
end
diag(DX)'

% Compute Corrl matrix and eigen values for Y samples
corY = zeros(nDim,nDim);
for i = 1 : nSamples
    corY = corY + y(:,i)*y(:,i)'/nSamples;
end
[V1,D1] = eig(corY);
[d,in] = sort(diag(D1));
DY = zeros(nDim,nDim);
for i = 1 : nDim
    DY(i,i) = D1(in(nDim+1-i),in(nDim+1-i));
end
diag(DY)'

% Compute true Generalized Eigenvectors
[V1,D1] = eig(corX, corY);
[d,in] = sort(diag(D1));
V = zeros(nDim,nDim);
D = zeros(nDim,nDim);
for i = 1 : nDim
    V(:,i) = V1(:,in(nDim+1-i)) / norm(V1(:,in(nDim+1-i)));
    D(i,i) = D1(in(nDim+1-i), in(nDim+1-i));
end
V(:,1:nEA)
diag(D)'

% Adaptive algorithm
A = zeros(nDim,nDim);
B = zeros(nDim,nDim);
W1 = 0.1 * ones(nDim,nEA);
W2 = W1;
W3 = W1;
C  = diag([2.6 2.3 2.0 1.7])%2.3-0.4*[1:nEA])
I  = eye(nDim,nDim);
for epoch = 1 : nEpochs
    for iter = 1 : nSamples
        
        cnt = nSamples*(epoch-1) + iter;
      
        A = A + (1.0/cnt)*((x(:,iter) * x(:,iter)') - A);
        B = B + (1.0/cnt)*((y(:,iter) * y(:,iter)') - B);
        etat1 = 1.0/(140 + cnt);
        etat2 = 1.0/(500 + cnt);
      
        % Homogeneous Gradient Descent
        %W1 = W1 + etat*(A*W1 - 0.5*B*W1*(W1'*A*W1) - 0.5*A*W1*(W1'*B*W1));
      
        % Deflated Gradient Descent
        W2 = W2 + etat1*(A*W2 - 0.5*B*W2*triu(W2'*A*W2) - 0.5*A*W2*triu(W2'*B*W2));
      
        % Weighted Gradient Descent
        W3 = W3 + etat2*(A*W3*C - 0.5*B*W3*C*(W3'*A*W3) - 0.5*A*W3*C*(W3'*B*W3));
      
        for i = 1 : nEA      
      	    %u1 = W1(:,i)/sqrt(W1(:,i)' * B * W1(:,i));
      	    u2 = W2(:,i)/sqrt(W2(:,i)' * W2(:,i));
      	    u3 = W3(:,i)/sqrt(W3(:,i)' * W3(:,i));
      	    %cos_t1(i,cnt) = abs(u1' * V(:,i));
            cos_t2(i,cnt) = abs(u2' * V(:,i));
            cos_t3(i,cnt) = abs(u3' * V(:,i));
        end
        ind(cnt) = cnt;
    end
end

if (1)

W2byNorm = W2*sqrt(inv(diag(diag(W2'*B*W2))))
V1to4 = V(:,1:nEA)
W3byNorm = W3*sqrt(inv(diag(diag(W3'*B*W3))))
V1to4 = V(:,1:nEA)

W2W2 = (W2'*B*W2)
W3W3 = (W3'*B*W3)

EstL2 = (diag(W2'*A*W2) ./ diag(W2'*B*W2))'
EstL3 = (diag(W3'*A*W3) ./ diag(W3'*B*W3))'
ActL  = diag(D(:,1:nEA))'

% Plot the convergence results

figure(1);
plot(ind,cos_t2(1,:),'k-'); hold on;
plot(ind,cos_t3(1,:),'k:'); hold on;
plot(0,1,'k:'); hold on;
legend('Deflated','Weighted',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 1');
hold off;

% Plot the convergence results
figure(2);
plot(ind,cos_t2(2,:),'k-'); hold on;
plot(ind,cos_t3(2,:),'k:'); hold on;
plot(0,1,'k:'); hold on;
legend('Deflated','Weighted',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 2');
hold off;

% Plot the convergence results
figure(3);
plot(ind,cos_t2(3,:),'k-'); hold on;
plot(ind,cos_t3(3,:),'k:'); hold on;
plot(0,1,'k:'); hold on;
legend('Deflated','Weighted',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 3');
hold off;

% Plot the convergence results
figure(4);
plot(ind,cos_t2(4,:),'k-'); hold on;
plot(ind,cos_t3(4,:),'k:'); hold on;
plot(0,1,'k:'); hold on;
legend('Deflated','Weighted',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 4');
hold off;

else

W1byNorm = W1*sqrt(inv(diag(diag(W1'*B*W1))))
V1to4 = V(:,1:nEA)
W2byNorm = W2*sqrt(inv(diag(diag(W2'*B*W2))))
V1to4 = V(:,1:nEA)
W3byNorm = W3*sqrt(inv(diag(diag(W3'*B*W3))))
V1to4 = V(:,1:nEA)

W1W1 = (W1'*B*W1)
W2W2 = (W2'*B*W2)
W3W3 = (W3'*B*W3)

EstL1 = (diag(W1'*A*W1) ./ diag(W1'*B*W1))'
EstL2 = (diag(W2'*A*W2) ./ diag(W2'*B*W2))'
EstL3 = (diag(W3'*A*W3) ./ diag(W3'*B*W3))'
ActL  = diag(D(:,1:nEA))'

% Plot the convergence results
figure(1);
plot(ind,cos_t1(1,:),'r:'); hold on;
plot(ind,cos_t2(1,:),'g-'); hold on;
plot(ind,cos_t3(1,:),'b--'); hold on;
legend('Symmetric','Deflated','Weighted',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 1');
hold off;

% Plot the convergence results
figure(2);
plot(ind,cos_t1(2,:),'r:'); hold on;
plot(ind,cos_t2(2,:),'g-'); hold on;
plot(ind,cos_t3(2,:),'b--'); hold on;
legend('Symmetric','Deflated','Weighted',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 2');
hold off;

% Plot the convergence results
figure(3);
plot(ind,cos_t1(3,:),'r:'); hold on;
plot(ind,cos_t2(3,:),'g-'); hold on;
plot(ind,cos_t3(3,:),'b--'); hold on;
legend('Symmetric','Deflated','Weighted',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 3');
hold off;

% Plot the convergence results
figure(4);
plot(ind,cos_t1(4,:),'r:'); hold on;
plot(ind,cos_t2(4,:),'g-'); hold on;
plot(ind,cos_t3(4,:),'b--'); hold on;
legend('Symmetric','Deflated','Weighted',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 4');
hold off;

end

%keyboard




  