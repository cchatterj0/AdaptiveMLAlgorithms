% This code is written to support the experiments in the book titled:
% Adaptive Machine Learning Algorithms with Python
% by
% Chanchal Chatterjee
% December 2021
% Solve Cubic Polynomial Roots

function CubicEqn();

a3 = 4; a2 = 3; a1 = 2; a0 = 1;
c = [a3 a2 a1 a0];
rts = roots(c)

b2 = a2/a3; 
b1 = a1/a3; 
b0 = a0/a3;

p = b1 - (b2^2)/3;
q = 2*(b2/3)^3 - b1*b2/3 + b0;

m3 = 1;
m2 = 0;
m1 = 0;
m0 = -((-q/2) + sqrt((q/2)^2 + (p/3)^3));
c = [m3 m2 m1 m0];
u = roots(c);
v = p ./ (3 * u);
z = u - v

a3*(z.^3) + a2*(z.^2) + a1*z + a0

m3 = 1;
m2 = 0;
m1 = 0;
m0 = -((-q/2) - sqrt((q/2)^2 + (p/3)^3));
c = [m3 m2 m1 m0];
u = roots(c);
v = p ./ (3 * u);
z = u - v

a3*(z.^3) + a2*(z.^2) + a1*z + a0

%keyboard