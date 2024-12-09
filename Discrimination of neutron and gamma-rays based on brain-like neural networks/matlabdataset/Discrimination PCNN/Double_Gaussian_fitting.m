function [B,C] = Double_Gaussian_fitting(n,t) 
%% Function of Double_Gaussian_fitting
    X = 1:length(n);
    q = 0.20 * length(n);
    [~,I] = max(n);
    L = fix(I-q);
    R = fix(I+q);
    if L<= 0
        L = 1;
    end
    bar(X,n);
    hold on
    center = X(L:R);
    counts = n(L:R); 
    duandian = T(n,t);
    [~,~,~,b,c] = Gauss(center,counts);
    for var1 = 1:duandian
        n(var1) = 0;
    end
    [~,I] = max(n);
    L= fix(I-q);
    R = fix(I+q);
    if R >= length(n)
        R = length(n);
    end
    center = X(L:R);
    counts = n(L:R); 
   [~,~,~,b2,c2] =Gauss(center,counts);  
   B(1) = b;
   C(1) = c;
   B(2) = b2;
   C(2) = c2;
  end
  
  function [position] = T(n,t)
    [~,I ]= max(n);
    I = I +t;
    var = I;
    m = length(n);
while 1
    y = n(var-t:var+t);
    x = var-t:var+t;
    xi = var-t:0.01:var+t;
    p= polyfit(x,y,3);
    yi = polyval(p,xi);
    if (diff(yi)>0)
        position = var-2*t;
        return;
    end
    if m <= var + t
        return ; 
    end
    var = var + 1;
end
end

function [ X,sigma,miu,a1,R,y_nihe ] = Eigenvalue( x_data,y_data )
X=zeros(size(x_data,2),3);
for i=1:size(x_data,2)
    for j=1:3
        X(i,j)=power(x_data(i),j-1);
    end
end
D = (inv(X'*X)*X')*y_data';
sigma = sqrt(-1/(2*D(3)));
miu = -D(2)/(2*D(3));
a1 = exp(D(1)+miu^2/(-1/D(3)));
for i=1:size(x_data,2)
    y_nihe(i,1)=a1*exp(-(x_data(1,i)-miu)^2/(-1/(2*D(3))));
end
l1=0;
l2=0;
l12=0;
for i=1:size(x_data,2)
    l1=l1+(y_data(i)-mean(y_data))^2;
    l2=l2+(y_nihe(i)-mean(y_nihe))^2;
    l12=l12+(y_nihe(i)-mean(y_nihe))*(y_data(i)-mean(y_data));
end
R=l12/sqrt(l1*l2);
end

function [x,y,fitresult,B,C] = Gauss(center,counts)
[~,sigma,c1,a1,~] = Eigenvalue(center,counts);
[xData, yData] = prepareCurveData( center, counts );
ft = fittype( 'gauss1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0];
opts.Robust = 'LAR';
opts.StartPoint = [a1 sigma c1];
[fitresult] = fit( xData, yData, ft, opts );
a1 = fitresult.a1;
b1 = fitresult.b1;
c1 = fitresult.c1;
C = c1;
B = b1;
left = b1 - 3*c1;
right = b1 + 3*c1;
x = left:0.001:right;
y = a1*exp(-((x-b1)/(c1)).^2);
plot(x,y,'red','LineWidth',2);
hold on
end
