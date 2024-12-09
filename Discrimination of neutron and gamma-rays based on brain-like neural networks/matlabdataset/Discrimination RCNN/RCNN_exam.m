function YY = RCNN_exam(I)
S = im2double(I);
[m,n]=size(S);YY=zeros(m,n);
U=YY;
E=YY+1;Y=YY;
B = 0.65;
V=1;
aT = 0.19;
vT =0.13;
aF=0.1;
t = 146;
d = 9;

% B = 0.4;
% V=1;
% aT = 0.709;
% vT =0.101;
% aF=0.205;
% t = 20;
% d = 9;
sigma1 = 4;sigma2 = 6;
W_default=fspecial('gaussian',d,sigma1);
W_default((d+1)/2,(d+1)/2)=0;
% Proceso iterativo
for i = 1:t            
    W = W_default.*rand_matrix(d,0.1,'norm',sigma2);
    L = conv2(Y,W,'same');
    U = U.*exp(-aF) + S.*(1+V*B*L);
    Y = im2double(U>E);             
    E = exp(-aT)*E + vT*Y;
    YY = YY+Y;
end
