% Adaptive SVD Algorithm 

function SVDPF();

clear all; 
close all; 

% Number of samples N and dimension D
eta0 = 200;
nSamples = 500;
nDim1 = 5;
nDim2 = 7;
nEpochs = 3;
randn('state',0);

% Generate the x random vectors
for i = 1 : nSamples
   for j = 1 : nDim1
      x(j,i) = (20 - 2*j) * randn;
   end
end
meanX = zeros(nDim1,1);
for i = 1 : nSamples
   meanX = meanX + x(:,i)/nSamples;
end
for i = 1 : nSamples
   x(:,i) = x(:,i) - meanX;
end

% Generate the y random vectors
for i = 1 : nSamples
   for j = 1 : nDim2
      y(j,i) = 3*randn;
   end
end
meanY = zeros(nDim2,1);
for i = 1 : nSamples
   meanY = meanY + y(:,i)/nSamples;
end
for i = 1 : nSamples
   y(:,i) = y(:,i) - meanY;
end

% Generate the Atrue matrix
Ctrue = zeros(nDim1,nDim2);
for i = 1 : nSamples
   Ctrue = Ctrue + x(:,i)*y(:,i)'/nSamples;
end

% Compute the true eigenvectors and eigenvalues
[U,D,V] = svd(Ctrue);
U
V
diag(D)'

% Run the adaptive algorithm
C  = zeros(nDim1,nDim2);
Z1 = zeros(nDim1,nDim1);
Z2 = zeros(nDim2,nDim2);
A  = zeros(nDim1+nDim2,nDim1+nDim2);
W  = 0.1 * ones(nDim1+nDim2,nDim2);

I  = eye(nDim2,nDim2);
mu = 2;

for epoch = 1 : nEpochs
   for iter = 1 : nSamples
      Ck = x(:,iter) * y(:,iter)';
      C = C + (1.0/(nSamples*(epoch-1) + iter))*(Ck - C);
      A = [Z1,C;C',Z2];
      etat = 1.0/(eta0 + nSamples*(epoch-1) + iter);
      W = W + etat*(A*W - mu*W*triu(W'*W - 2*I));
      
      U1 = W(1:nDim1,1:nDim1);
      for i = 1 : nDim1
        a = norm(U1(:,i));
   		U1(:,i) = U1(:,i)/a;
      end
      V1 = W(nDim1+1:nDim1+nDim2,1:nDim2);
      for i = 1 : nDim2
        a = norm(V1(:,i));
   		V1(:,i) = V1(:,i)/a;
      end
      cos_t1(:,nSamples*(epoch-1) + iter) = abs(diag(U1'*U));
      cos_t2(:,nSamples*(epoch-1) + iter) = abs(diag(V1'*V));
   end
end

% Check the final norms
U1 = W(1:nDim1,1:nDim1);
U1byNorm = U1*sqrt(inv(diag(diag(U1'*U1))));
for i = 1 : nDim1
   UNorm(i) = norm(U1(:,i));
end
V1 = W(nDim1+1:nDim1+nDim2,1:nDim2);
V1byNorm = V1*sqrt(inv(diag(diag(V1'*V1))));
for i = 1 : nDim2
   VNorm(i) = norm(V1(:,i));
end
UNorm
VNorm
U1byNorm
V1byNorm
dd = diag(U1byNorm'*C*V1byNorm);
dd'

for iter = 1 : nSamples*nEpochs
   ind(iter) = iter;
end

if (1)

% Plot the convergence results
figure(1);
plot(ind,cos_t1(1,:),'k-'); hold on;
plot(ind,cos_t1(2,:),'k:'); hold on;
plot(ind,cos_t1(3,:),'k--'); hold on;
legend('Component1 of U','Component2 of U','Component3 of U',0);
xlabel('Number of Samples');
ylabel('Direction Cosines for Components of U');
hold off;

% Plot the convergence results
figure(2);
plot(ind,cos_t2(1,:),'k-'); hold on;
plot(ind,cos_t2(2,:),'k:'); hold on;
plot(ind,cos_t2(3,:),'k--'); hold on;
legend('Component1 of V','Component2 of V','Component3 of V',0);
xlabel('Number of Samples');
ylabel('Direction Cosines for Components of V');
hold off;


else

plot(ind,cos_t1(1,:),'k-'); hold on;
plot(ind,cos_t1(2,:),'k:');
plot(ind,cos_t1(3,:),'k-.');
plot(ind,cos_t1(4,:),'k--');

plot(ind,cos_t2(1,:),'r-');
plot(ind,cos_t2(2,:),'r:');
plot(ind,cos_t2(3,:),'r-.');
plot(ind,cos_t2(4,:),'r--');
hold off;

end





  