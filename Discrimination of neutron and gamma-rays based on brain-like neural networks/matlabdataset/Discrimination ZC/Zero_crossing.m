function R = Zero_crossing(DATA)
%% The main programme of the Zero-crossing method
% Reference:https://doi.org 10.1109/TNS.2011.2164556
% signal = textread("nomalizedDATA.txt");% Read radiation pulse signals
%% Zero-crossing calculation
T = 0.0000000001;
constant = 0.000000007;
a=1/constant;
alpha=exp(-T/constant); 
Num_signal=size(DATA,1);
filtered_DATA = DATA;
parfor i=1:Num_signal
data_filtered = filtered_DATA(i,:);
L=length(data_filtered);
x=[0,0,0,data_filtered];y=zeros(1,L+3);data1=[];
for n=4:1:L+3
    y(n)=3*alpha*(y(n-1))-3*(alpha^2)*(y(n-2))+(alpha^3)*(y(n-3))+T*alpha*(1-(a*T/2))*(x(n-1))-T*(alpha^2)*(1+(a*T/2))*(x(n-2));
    data1(n)=y(n);
end
[~,maxposition] = max(data1);
for j=maxposition:1:L
    if data1(j)<0
            stoppoint=j;
            break
    end
end
 startpoint=maxposition*0.1;
 startpoint=round(startpoint);
 R(i)=(stoppoint-startpoint);
end
% %% Figure of Merit Calculation
% R = mapminmax(R,0,50);
% [n,~] = hist(R,50);
% [miu,sigma] = Double_Gaussian_fitting(n,3);
% FOM = (miu(2)-miu(1))/(1.667*(sigma(2)+sigma(1)));
% str_FOM = sprintf('%f',FOM);
% str =['Gauss FOM =',str_FOM];
% dim = [.40 .50 .3 .3];
% annotation('textbox',dim,'String',str,'FontSize',12,'FitBoxToText','on');
end



