% Mean and Median 

function MeanMedian;

clear all; 
close all; 

% Number of samples N and dimension D
nSamples = 500;
nEpochs  = 1;
randn('state',0);

% Generate the covX random matrices
if 1
nDim     = 5;
mX = [10 7 6 5 1]';
covX = ...
[  2.091  0.038 -0.053 -0.005  0.010 ; ...
   0.038  1.373  0.018 -0.028 -0.011 ; ...
  -0.053  0.018  1.430  0.017  0.055 ; ...
  -0.005 -0.028  0.017  1.084 -0.005 ; ...
   0.010 -0.011  0.055 -0.005  1.071];
else
nDim     = 10;
mX = [10 9 8 7 6 5 4 3 2 1]';
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
end

% Compute data
[V1,D1] = eig(covX);
[d,in]  = sort(diag(D1));
V = zeros(nDim,nDim);
D = zeros(nDim,nDim);
for i = 1 : nDim
    V(:,i) = V1(:,in(nDim+1-i));
    D(i,i) = D1(in(nDim+1-i),in(nDim+1-i));
end
for i = 1 : nSamples
    x(1:nDim,i) = V * sqrt(D)*(randn(nDim,1) + mX);
end

% Compute mean, median, norm mean
Var      = zeros(nDim,nDim);
for i = 1 : nSamples
    Var  = Var  + x(:,i)*x(:,i)';
end
Var1      = Var / nSamples;
Mean1     = Mean(x')';
NMean1    = Mean1 / norm(Mean1);
Median1   = Median(x')';

% Adaptive algorithm
A  = zeros(nDim,nDim);
m  = zeros(nDim,1);
md = zeros(nDim,1);
w1 = zeros(nDim,1); %nm = nm / norm(nm);
w2 = zeros(nDim,1); %nm = nm / norm(nm);
w3 = zeros(nDim,1); %nm = nm / norm(nm);
mu = 150;

for epoch = 1 : nEpochs
    for iter = 1 : nSamples
        cnt = nSamples*(epoch-1) + iter;

        A  = A  + (1.0/cnt)*((x(:,iter) * x(:,iter)') - A);
        m  = m  + (1.0/cnt)*(x(:,iter) - m);
        md = md + (3.0/cnt)*sign(x(:,iter) - md);
        
        w1 = w1 + (1.0/(100+cnt))*(x(:,iter) - w1'*x(:,iter)*w1);
        w2 = w2 + (1.0/(100+cnt))*(x(:,iter) - w2 - mu*w2*(w2'*w2 - 1));
        w3 = w3 + (1.0/(100+cnt))*(2*x(:,iter) - w3'*x(:,iter)*w3 - w3'*w3*x(:,iter));
        
        err1(cnt) = norm(A - Var1, 'fro');
        err2(cnt) = norm(m - Mean1);
        err3(cnt) = norm(md - Median1);
        err4(cnt) = norm(w1- NMean1);
        err5(cnt) = norm(w2- NMean1);
        err6(cnt) = norm(w3- NMean1);
        ind(cnt)  = cnt;
    end
end


% Plot the convergence results
figure(1);
plot(ind, err3,'k-'); hold on;
legend('Median by Alg. (2.50)',0);
xlabel('Number of Samples');
ylabel('Error = ||w_k - Actual Median||');
hold off;

figure(2);
plot(ind, err4,'k-'); hold on;
legend('Normalized Mean by Alg. (2.39)',0);
xlabel('Number of Samples');
ylabel('Error = ||w_k - Actual Normalized Mean||');
hold off;

figure(3);
plot(ind, err5,'k-'); hold on;
legend('Normalized Mean by Alg. (2.46)',0);
xlabel('Number of Samples');
ylabel('Error = ||w_k - Actual Normalized Mean||');
hold off;

figure(4);
plot(ind, err6,'k-'); hold on;
legend('Normalized Mean by Alg. (2.44)',0);
xlabel('Number of Samples');
ylabel('Error = ||w_k - Actual Normalized Mean||');
hold off;

norm(md - Median1)
norm(w1- NMean1)
norm(w2- NMean1)
norm(w3- NMean1)
