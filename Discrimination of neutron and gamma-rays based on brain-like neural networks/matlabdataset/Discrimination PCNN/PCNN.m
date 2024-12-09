function TS = PCNN(I)
%% Function of PCNN
[m,n] = size(I);
S = im2double(I);
l = 0.1091;
r = 0.1409;
W = [l r l;
     r 0 r;
     l r l];
Y = zeros(m,n); U = Y;TS=Y;
% k =180;
k =31;
M = W;
F = Y; L = Y; U = Y; 
E = Y;
% al=0.356;vl=0.0005;ve=15.5;ae=0.081;af=0.325;vf=0.0005;beta=0.67;
al=0.356;vl=0.0005;ve=5.0;ae=0.13;af=0.325;vf=0.0005;beta=0.67;

for t=1:k
    F = exp(-af) *F +vf*conv2(Y,M,'same') + S;  
	L = exp(-al) *L +vl*conv2(Y,W,'same');              
	U = F.*(1 + beta*L);
    % Y = double(U>E)
    X = 1./(1+exp(E - U));
    Y = double(X > 0.5);
    E = exp(-ae) * E + ve * Y;
    TS =TS+Y;
end