% Adaptive GSVD Algorithm 

function GSVDOJA();

% Number of samples N and dimension D
eta0     = 500;
nSamples = 500;
nDim1    = 5;
nDim2    = 7;
nEpochs  = 3;
randn('state',0);

% Generate the x random vectors
for i = 1 : nSamples
    for j = 1 : nDim1
        x(j,i) = 8*(20-2*j)*randn;
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
        y(j,i) = 1*randn;
    end
end
meanY = zeros(nDim2, 1);
for i = 1 : nSamples
    meanY = meanY + y(:,i)/nSamples;
end
for i = 1 : nSamples
    y(:,i) = y(:,i) - meanY;
end

% Generate the Ctrue matrix
Ctrue = zeros(nDim1, nDim2);
for i = 1 : nSamples
    Ctrue = Ctrue + x(:,i)*y(:,i)'/nSamples;
end
Ctrue

% Generate the Etrue matrix
Etrue = zeros(nDim2, nDim2);
for i = 1 : nSamples
    Etrue = Etrue + y(:,i)*y(:,i)' / nSamples;
end
Etrue

% Compute the true eigenvectors and eigenvalues
Z1  = zeros(nDim1,nDim1);
Z2  = zeros(nDim2,nDim2);
Z3  = zeros(nDim1,nDim2);
I1  = eye(nDim1,nDim1);
At   = zeros(nDim1+nDim2,nDim1+nDim2);
Bt   = zeros(nDim1+nDim2,nDim1+nDim2);
At   = [Z1,Ctrue; Ctrue',Z2];
Bt   = 0.5 * [I1,Z3; Z3',Etrue];
[Vx,Dx] = eig(At, Bt);
[d,in]  = sort(diag(Dx));
Vy  = zeros(nDim1+nDim2,nDim1+nDim2);
Dy  = zeros(nDim1+nDim2,nDim1+nDim2);
for i = 1 : nDim1
    r = in(nDim1+nDim2+1-i);
    Vy(:,i) = Vx(:,r);
    Dy(i,i) = Dx(r,r);
end
U = Vy(1:nDim1,1:nDim1);
V = Vy(nDim1+1:nDim1+nDim2, 1:nDim1);
for i = 1 : nDim1
    U(:,i) = U(:,i) / norm(U(:,i));
    V(:,i) = V(:,i) / norm(V(:,i));
end
diag(Dy)'

%[Ua,Va,Xa,Da,Sa] = GSVD(Ctrue, Etrue);
%SV     = GSVD(Ctrue, Etrue)'
%for i = 1 : nDim1
%    Ua(:,i) = Ua(:,i) / norm(Ua(:,i));
%    Va(:,i) = Va(:,i) / norm(Va(:,i));
%    Xa(:,i) = Xa(:,i) / norm(Xa(:,i));
%end


% Run the adaptive algorithm
C   = zeros(nDim1,nDim2);
E   = zeros(nDim2,nDim2);
A   = zeros(nDim1+nDim2,nDim1+nDim2);
B   = zeros(nDim1+nDim2,nDim1+nDim2);
W   = 0.1 * ones(nDim1+nDim2,nDim2);

if 1
for epoch = 1 : nEpochs
    for iter = 1 : nSamples
        
        cnt = nSamples*(epoch-1) + iter;
        
        Ck   = x(:,iter) * y(:,iter)';
        C    = C + (1.0 / cnt)*(Ck - C);
        A   = [Z1,C; C',Z2];
        
        Ek   = y(:,iter) * y(:,iter)';
        E    = E + (1.0/cnt)*(Ek - E);
        B   = 0.5 * [I1,Z3; Z3',E];
       
        etat = 1.0 / (eta0 + cnt);
        W    = W + etat*(A*W - B*W*triu(W'*A*W));
      
        U1 = W(1:nDim1,1:nDim1);
        for i = 1 : nDim1
   		    U1(:,i) = U1(:,i)/norm(U1(:,i));
        end
        V1 = W(nDim1+1:nDim1+nDim2, 1:nDim1);
        for i = 1 : nDim1
   		    V1(:,i) = V1(:,i)/norm(V1(:,i));
        end
        cos_t1(:,cnt) = abs(diag(U1'*U));
        cos_t2(:,cnt) = abs(diag(V1'*V));
        ind(cnt) = cnt;
    end
end

if (1)

% Plot the convergence results
figure(1);
plot(ind,cos_t1(1,:),'k-'); hold on;
plot(ind,cos_t1(2,:),'k:'); hold on;
plot(ind,cos_t1(3,:),'k--'); hold on;
plot(0,1,'k:'); hold on;
legend('Component1 of U','Component2 of U','Component3 of U',0);
xlabel('Number of Samples');
ylabel('Direction Cosines for Components of U');
hold off;

% Plot the convergence results
figure(2);
plot(ind,cos_t2(1,:),'k-'); hold on;
plot(ind,cos_t2(2,:),'k:'); hold on;
plot(ind,cos_t2(3,:),'k--'); hold on;
plot(0,1,'k:'); hold on;
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

end