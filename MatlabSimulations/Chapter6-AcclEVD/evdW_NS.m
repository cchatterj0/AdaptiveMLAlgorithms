% Speedup of the Adaptive EVD Algorithm 
% Compare Our Steepest Descent, Yang's PASTd, Bannour/Sadjadi's RLS, Fu/Dowling's CGET1
% Nonstationary Case

function evdW;

clear all; close all; 

% Number of samples N and dimension D
nSamples1 = 500;
nSamples2 = 1000;
nSamples = nSamples1 + nSamples2;
nDim = 10;
nEA = 4;
nEpochs = 1;
randn('state',0);

% Generate the covX random matrices
if 0
covX = ...
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

covX = ...
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

covX = ...
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

covX = ...
[ 0.0840  -0.0120   0.0010   0.0150  -0.0040   0.0240  -0.0110   0.0270   0.0730  -0.0120 ; ...
 -0.0120   0.7930   0.0000   0.1970   0.0310  -0.0260   0.0500   0.0790   0.1450  -0.0270 ; ...
  0.0010   0.0000   0.0660  -0.0230   0.0110  -0.0170   0.0390  -0.0020   0.0170  -0.0040 ; ...
  0.0150   0.1970  -0.0230   0.7280  -0.0030  -0.0150   0.1210   0.0900   0.0800   0.0410 ; ...
 -0.0040   0.0310   0.0110  -0.0030   0.0800  -0.0010  -0.0070   0.0230  -0.0140  -0.0230 ; ...
  0.0240  -0.0260  -0.0170  -0.0150  -0.0010   0.3360  -0.1470  -0.0370   0.3020  -0.0520 ; ...
 -0.0110   0.0500   0.0390   0.1210  -0.0070  -0.1470   1.4500  -0.1320  -0.2300  -0.1370 ; ... 
  0.0270   0.0790  -0.0020   0.0900   0.0230  -0.0370  -0.1320   0.3200   0.2380   0.0170 ; ...
  0.0730   0.1450   0.0170   0.0800  -0.0140   0.3020  -0.2300   0.2380   2.9300  -0.0570 ; ...
 -0.0120  -0.0270  -0.0040   0.0410  -0.0230  -0.0520  -0.1370   0.0170  -0.0570   0.3040 ];

covX = ...
[ 0.0900   0.0010  -0.0080  -0.1910  -0.0070   0.0410  -0.0300  -0.0580   0.0220   0.0320 ; ...
  0.0010   0.0920  -0.0110   0.0080  -0.0140  -0.0020   0.0120  -0.0100  -0.0210  -0.0020 ; ...
 -0.0080  -0.0110   0.0820   0.0820   0.0140  -0.0200  -0.0580   0.1050   0.0040   0.0230 ; ...
 -0.1910   0.0080   0.0820   5.6800  -0.0960  -0.0150   0.6460   0.2190  -0.2380   0.2180 ; ...
 -0.0070  -0.0140   0.0140  -0.0960   0.0760  -0.0350  -0.0400  -0.0230   0.0270  -0.0140 ; ...
  0.0410  -0.0020  -0.0200  -0.0150  -0.0350   0.4580   0.1380  -0.2510   0.0120   0.0390 ; ...
 -0.0300   0.0120  -0.0580   0.6460  -0.0400   0.1380   1.8200  -0.1830  -0.0020   0.1170 ; ...
 -0.0580  -0.0100   0.1050   0.2190  -0.0230  -0.2510  -0.1830   4.0700  -0.4640   0.1470 ; ...
  0.0220  -0.0210   0.0040  -0.2380   0.0270   0.0120  -0.0020  -0.4640   0.2630   0.0540 ; ...
  0.0320  -0.0020   0.0230   0.2180  -0.0140   0.0390   0.1170   0.1470   0.0540   0.3870 ];

covX = 20*[0.091 0.038 -.053 -.005 0.010; ...
           0.038 0.373 0.018 -.028 -.011; ...
           -.053 0.018 1.430 0.017 0.055; ...
           -.005 -.028 0.017 0.084 -.005; ...
           0.010 -.011 0.055 -.005 0.071];
end

covX1 = 2*...
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
covX2 = 4* ...
[ 0.0900   0.0010  -0.0080  -0.1910  -0.0070   0.0410  -0.0300  -0.0580   0.0220   0.0320 ; ...
  0.0010   0.0920  -0.0110   0.0080  -0.0140  -0.0020   0.0120  -0.0100  -0.0210  -0.0020 ; ...
 -0.0080  -0.0110   0.0820   0.0820   0.0140  -0.0200  -0.0580   0.1050   0.0040   0.0230 ; ...
 -0.1910   0.0080   0.0820   5.6800  -0.0960  -0.0150   0.6460   0.2190  -0.2380   0.2180 ; ...
 -0.0070  -0.0140   0.0140  -0.0960   0.0760  -0.0350  -0.0400  -0.0230   0.0270  -0.0140 ; ...
  0.0410  -0.0020  -0.0200  -0.0150  -0.0350   0.4580   0.1380  -0.2510   0.0120   0.0390 ; ...
 -0.0300   0.0120  -0.0580   0.6460  -0.0400   0.1380   1.8200  -0.1830  -0.0020   0.1170 ; ...
 -0.0580  -0.0100   0.1050   0.2190  -0.0230  -0.2510  -0.1830   4.0700  -0.4640   0.1470 ; ...
  0.0220  -0.0210   0.0040  -0.2380   0.0270   0.0120  -0.0020  -0.4640   0.2630   0.0540 ; ...
  0.0320  -0.0020   0.0230   0.2180  -0.0140   0.0390   0.1170   0.1470   0.0540   0.3870 ];

    
% Compute data
[V1,D1] = eig(covX1);
[d1,in1] = sort(diag(D1));
[V2,D2]= eig(covX2);
[d2,in2]=sort(diag(D2));
VV1 = zeros(nDim,nDim);
VV2 = zeros(nDim,nDim);
DD1 = zeros(nDim,nDim);
DD2 = zeros(nDim,nDim);
for i = 1 : nDim
   VV1(:,i) = V1(:,in1(nDim+1-i));
   VV2(:,i) = V2(:,in2(nDim+1-i));
   DD1(i,i) = D1(in1(nDim+1-i),in1(nDim+1-i));
   DD2(i,i) = D2(in2(nDim+1-i),in2(nDim+1-i));
end
x1(1:nDim,1:nSamples1) = VV1*sqrt(DD1)*randn(nDim,nSamples1);
x2(1:nDim,1:nSamples2) = VV2*sqrt(DD2)*randn(nDim,nSamples2);

% Compute Corrl matrix and eigen vectors
corX1 = zeros(nDim,nDim);
corX2 = zeros(nDim,nDim);
for i = 1 : nSamples1
   corX1 = corX1 + x1(:,i)*x1(:,i)'/nSamples1;
end
for i = 1 : nSamples2
   corX2 = corX2 + x2(:,i)*x2(:,i)'/nSamples2;
end
[V1,D1] = eig(corX1);
[d1,in1] = sort(diag(D1));
[V2,D2] = eig(corX2);
[d2,in2] = sort(diag(D2));

VV1 = zeros(nDim,nDim);
VV2 = zeros(nDim,nDim);
DD1 = zeros(nDim,nDim);
DD2 = zeros(nDim,nDim);
for i = 1 : nDim
   VV1(:,i) = V1(:,in1(nDim+1-i));
   VV2(:,i) = V2(:,in2(nDim+1-i));
   DD1(i,i) = D1(in1(nDim+1-i),in1(nDim+1-i));
   DD2(i,i) = D2(in2(nDim+1-i),in2(nDim+1-i));
end

%corX1
%V1
diag(DD1)'
%corX2
%V2
diag(DD2)'


% Adaptive algorithm

% Initial Values for Steepest Descent and Xu's Alg.
Weight = 0.995;
A = zeros(nDim,nDim);
W1 = 0.1 * ones(nDim,nEA);
I  = eye(nDim,nDim);

% Initial Values for PASTd
W2 = 0.1 * ones(nDim,nEA);
d = 0.2*ones(1,nEA);
y = zeros(1,nEA);
E = zeros(nDim,nEA);
XX = zeros(nDim,nEA+1);
beta = Weight;

% Initial Values of RLS - Bannour/Sadjadi
W3 = 0.1 * ones(nDim,nEA);
h = zeros(1,nEA);
K = zeros(1,nEA);
P = ones(1,nEA);
dd = zeros(nDim,1);

% Initial Values of CGET1
W4 = 0.1*ones(nDim,nDim);
lambda = zeros(1,nEA);
Residue = zeros(nDim,nEA);
PP = zeros(nDim,nEA);
XX = zeros(nDim,nEA);
for i = 1 : nEA
   A = x1(:,1) * x1(:,1)';
   lambda(i) = W4(:,i)' * A * W4(:,i);
   Residue(:,i) = lambda(i)*W4(:,i) - A*W4(:,i);
   PP(:,i) = Residue(:,i)/norm(Residue(:,i));
end

for epoch = 1 : nEpochs
   for iter = 1 : nSamples
      
      if iter <= nSamples1
         Ak = x1(:,iter) * x1(:,iter)';
         xs = x1(:,iter);
      else 
         Ak = x2(:,iter-nSamples1) * x2(:,iter-nSamples1)';
         xs = x2(:,iter-nSamples1);
      end
      A = Weight * A + (1.0/(nSamples*(epoch-1) + iter))*(Ak - Weight * A);

      % Steepest Descent
      G = (-2*A*W1 + A*W1*triu(W1'*W1) + W1*triu(W1'*A*W1));
      for i = 1 : nEA
         g = G(:,i);
         w = W1(:,i);
         M = zeros(nDim,nDim);
         for k = 1 : i-1
            M = M + (A*W1(:,k)*W1(:,k)' + W1(:,k)*W1(:,k)'*A);
         end
         
         F = - 2*A + 2*A*w*w' + 2*w*w'*A + A*(w'*w) + (w'*A*w)*I  +  M;
      	a0 = g'*g;
      	a1 = - g'*F*g;
      	a2 = 3*((w'*A*g)*(g'*g) + (g'*A*g)*(w'*g));
      	a3 = - 2*(g'*A*g)*(g'*g);
      	c = [a3 a2 a1 a0];
         rts = roots(c);
         
         clear rs; clear r; clear J;
         cnt = 1; 
         for k = 1 : 3
            if isreal(rts(k))
               rs(cnt) = rts(k);
            	r = w - rts(k)*g;
               J(cnt) = (-2*r'*A*r + r'*A*r*r'*r + r'*M*r);
               cnt = cnt + 1;
            end
         end
         [yy, iyy] = min(J);
         alpha = rs(iyy);
         
         w = w - alpha * g;
		  	W1(:,i) = w;
        end
        
      % PASTd - Yang
      XX(:,1) = xs;
      for i = 1 : nEA
         y(i) = W2(:,i)' * XX(:,i);
         d(i) = beta * d(i) + y(i)^2;
         E(:,i) = XX(:,i) - W2(:,i)*y(i);
         W2(:,i) = W2(:,i) + E(:,i)*(y(i)/d(i));
         XX(:,i+1) = XX(:,i) - W2(:,i)*y(i);
      end
      
      % RLS - Bannour/Sadjadi
      for i = 1 : nEA
         h(i) = W3(:,i)' * xs;
         K(i) = (P(i)*h(i)) / (1 + h(i)^2 * P(i));
			dd = zeros(nDim,1);
         for j = 1 : i-1
            dd = dd + h(j)*W3(:,j);
         end
         W3(:,i) = W3(:,i) + K(i)*(xs - dd - h(i)*W3(:,i));
         P(i) = (1-K(i)*h(i))*P(i);
      end
      
      % CGET1 - Fu/Dowling
      for i = 1 : nEA
         if (PP(:,i)'*A*PP(:,i)) == 0
            iter
            i
            (PP(:,i)'*A*PP(:,i))
            keyboard
         end
         qq(i) = (Residue(:,i)'*A*PP(:,i)) / (PP(:,i)'*A*PP(:,i));
         PP(:,i) = Residue(:,i) - qq(i)*PP(:,i);
         DD = zeros(nDim,nDim);
         for j = 1 : i-1
            DD = DD + W4(:,j)*W4(:,j)';
         end
         BB = I - DD;
         AA = BB'*A*BB;
         lx = W4(:,i)'*AA*W4(:,i);
         ax = W4(:,i)'*AA*PP(:,i);
         bx = PP(:,i)'*AA*PP(:,i);
         ex = W4(:,i)'*BB*W4(:,i);
         cx = W4(:,i)'*BB*PP(:,i);
         dx = PP(:,i)'*BB*PP(:,i);
         fx = bx*ex - lx*dx;
         gx = ax*ex - lx*cx;
         hx = bx*cx - ax*dx;
         if hx == 0
            hx
            keyboard
         end
         tx = (-fx - sqrt(fx^2 - 4*hx*gx))/(2*hx);
         XX(:,i) = W4(:,i) + tx*PP(:,i);
         W4(:,i) = BB*XX(:,i)/sqrt(XX(:,i)'*BB*XX(:,i));
         lambda(i) = W4(:,i)'*A*W4(:,i);
         Residue(:,i) = lambda(i)*W4(:,i)-A*W4(:,i);
      end
      
      
      for i = 1 : nEA      
      	u1 = W1(:,i)/norm(W1(:,i));
      	u2 = W2(:,i)/norm(W2(:,i));
      	u3 = W3(:,i)/norm(W3(:,i));
      	u4 = W4(:,i)/norm(W4(:,i));
         if iter <= nSamples1
            cos_t1(i,iter) = abs(u1'*VV1(:,i));
            cos_t2(i,iter) = abs(u2'*VV1(:,i));
            cos_t3(i,iter) = abs(u3'*VV1(:,i));
            cos_t4(i,iter) = abs(u4'*VV1(:,i));
         else 
            cos_t1(i,iter) = abs(u1'*VV2(:,i));
            cos_t2(i,iter) = abs(u2'*VV2(:,i));
            cos_t3(i,iter) = abs(u3'*VV2(:,i));
            cos_t4(i,iter) = abs(u4'*VV2(:,i));
         end   
      end
      ind(iter) = iter;
   end
end


% Plot the convergence results
figure(1);
plot(ind,cos_t1(1,:),'k-'); hold on;
plot(ind,cos_t2(1,:),'k:'); hold on;
plot(ind,cos_t3(1,:),'k--'); hold on;
plot(ind,cos_t4(1,:),'k-.'); hold on;
legend('Steepest Descent','PASTd','RLS','CGET1',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 1');
hold off;

% Plot the convergence results
figure(2);
plot(ind,cos_t1(2,:),'k-'); hold on;
plot(ind,cos_t2(2,:),'k:'); hold on;
plot(ind,cos_t3(2,:),'k--'); hold on;
plot(ind,cos_t4(2,:),'k-.'); hold on;
legend('Steepest Descent','PASTd','RLS','CGET1',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 2');
hold off;

% Plot the convergence results
figure(3);
plot(ind,cos_t1(3,:),'k-'); hold on;
plot(ind,cos_t2(3,:),'k:'); hold on;
plot(ind,cos_t3(3,:),'k--'); hold on;
plot(ind,cos_t4(3,:),'k-.'); hold on;
legend('Steepest Descent','PASTd','RLS','CGET1',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 3');
hold off;

% Plot the convergence results
figure(4);
plot(ind,cos_t1(4,:),'k-'); hold on;
plot(ind,cos_t2(4,:),'k:'); hold on;
plot(ind,cos_t3(4,:),'k--'); hold on;
plot(ind,cos_t4(4,:),'k-.'); hold on;
legend('Steepest Descent','PASTd','RLS','CGET1',0);
xlabel('Number of Samples');
ylabel('Direction Cosine - Component 4');
hold off;


%keyboard




  