% A^(-1/2) - Comparison of Adaptive Square Root of A Algorithms

function A2;

clear all;
close all;

% Number of samples N and dimension D
nSamples = 500;
nDim     = 10;
nEpochs  = 1;
randn('state',0);

% Generate the covX random matrices
covX = 3*...
[ 0.0910   0.0380  -0.0530  -0.0050   0.0100  -0.1360   0.1550   0.0300   0.0020   0.0320 ; ...
  0.0380   0.3730   0.0180  -0.0280  -0.0110  -0.3670   0.1540  -0.0570  -0.0310  -0.0650 ; ...
 -0.0530   0.0180   1.4300   0.0170   0.0550  -0.4500  -0.0380  -0.2980  -0.0410  -0.0300 ; ...
 -0.0050  -0.0280   0.0170   0.0840  -0.0050   0.0160   0.0420  -0.0220   0.0010   0.0050 ; ...
  0.0100  -0.0110   0.0550  -0.0050   0.0710   0.0880   0.0580  -0.0690  -0.0080   0.0030 ; ...
 -0.1360  -0.3670  -0.4500   0.0160   0.0880   5.7200  -0.5440  -0.2480   0.0050   0.0950 ; ...
  0.1550   0.1540  -0.0380   0.0420   0.0580  -0.5440   2.7500  -0.3430  -0.0110  -0.1200 ; ...
  0.0300  -0.0570  -0.2980  -0.0220  -0.0690  -0.2480  -0.3430   1.4500   0.0780   0.0280 ; ...
  0.0020  -0.0310  -0.0410   0.0010  -0.0080   0.0050  -0.0110   0.0780   0.0670   0.0150 ; ...
  0.0320  -0.0650  -0.0300   0.0050   0.0030   0.0950  -0.1200   0.0280   0.0150   0.3410 ];

% Compute data
[V1,D1] = eig(covX);
[d,in] = sort(diag(D1));
V = zeros(nDim,nDim);
D = zeros(nDim,nDim);
for i = 1 : nDim
    V(:,i) = V1(:,in(nDim+1-i));
    D(i,i) = D1(in(nDim+1-i),in(nDim+1-i));
end
x(1:nDim, 1:nSamples) = V * sqrt(D) * randn(nDim,nSamples);

% Compute Corrl matrix and eigen vectors
corX = zeros(nDim,nDim);
for i = 1 : nSamples
    corX = corX + x(:,i)*x(:,i)' / nSamples;
end
[V1,D1] = eig(corX);
[d,in] = sort(diag(D1));
V = zeros(nDim,nDim);
D = zeros(nDim,nDim);
for i = 1 : nDim
    V(:,i) = V1(:,in(nDim+1-i));
    D(i,i) = D1(in(nDim+1-i),in(nDim+1-i));
end
corX
diag(D)'
Ah = V * sqrt(inv(D)) * V';

% Adaptive algorithm
A  = zeros(nDim,nDim);
W1 = eye(nDim,nDim);
W2 = W1;
W3 = W1;
I  = eye(nDim,nDim);

for epoch = 1 : nEpochs
    for iter = 1 : nSamples
        cnt = nSamples*(epoch-1) + iter;
      
        A = A + (1.0/cnt)*((x(:,iter) * x(:,iter)') - A);
        etat1 = 1.0/(20 + cnt);
        etat2 = 1.0/(30 + cnt);
      
        % Algorithm 1
        W1 = W1 + etat1 * (W1 - W1*W1'*A*W1);
      
        % Algorithm 2
        W2 = W2 + etat1 * (W2 - W2*A*W2'*W2);
      
        % Algorithm 3
        W3 = W3 + etat2 * (I - W3*A*W3);
      
        err1(cnt) = norm(I - W1'*A*W1, 'fro');
        err2(cnt) = norm(I - W2*A*W2', 'fro');
        err3(cnt) = norm(I - W3*A*W3, 'fro');
        err4(cnt) = norm(Ah - W3, 'fro');
        ind(cnt)  = cnt;
    end
end

% Plot the convergence results
figure(1);
plot(ind, err1, 'k-'); hold on;
legend('Method 1 - Equation (3.32)',0);
xlabel('Number of Samples');
ylabel('Error = ||I - W^TAW||');
hold off;

% Plot the convergence results
figure(2);
plot(ind, err2, 'k-'); hold on;
legend('Method 2 - Equation (3.44)',0);
xlabel('Number of Samples');
ylabel('Error = ||I - WAW^T||');
hold off;

% Plot the convergence results
figure(3);
plot(ind, err3, 'k-'); hold on;
legend('Method 3 - Equation (3.48)',0);
xlabel('Number of Samples');
ylabel('Error = ||I - WAW||');
hold off;


% Plot the convergence results
figure(4);
plot(ind, err4, 'k-'); hold on;
legend('Method 3 - Equation (3.48)',0);
xlabel('Number of Samples');
ylabel('Error = ||A^-^1^/^2 - W||');
hold off;

norm(I - W1'*A*W1, 'fro')
norm(I - W2*A*W2', 'fro')
norm(I - W3*A*W3, 'fro')
norm(Ah - W3, 'fro')

  