function R = PCNN_main(DATA)
%% The main programme of the PCNN
% Reference: https://doi.org/10.1007/s41365-021-00915-w
% signal = textread("nomalizedDATA.txt"); % Read radiation pulse signals

%% PCNN calculation
Num_signal=size(DATA,1);
R=zeros(1,Num_signal); % R is the discrimination factor
% DATA = filter_my(DATA,PARAM);
parfor i=1:Num_signal
    data = DATA(i,:);
    Ignition_map=PCNN(data);
    [maxvalue,maxposition] = max(data); % Find the maximum position of a pulse signal
    n0=7;
    m0=maxposition-n0;
    n2=200;
    SUM=sum(Ignition_map(m0:maxposition+n2));
    R(i)=SUM;
end
%% Figure of Merit Calculation
% R=mapminmax(R,0,1);
% R=R*200;  
% Max=max(R);Min=min(R);
% bins = Max-Min+1;
% [n,~] = hist(R,bins);
% [miu,sigma] = Double_Gaussian_fitting(n,5);
% FOM = (miu(2)-miu(1))/(1.667*(sigma(2)+sigma(1)));
% str_FOM = sprintf('%f',FOM);
% str =['FOM =',str_FOM];
% dim = [.75 .60 .3 .3];
% annotation('textbox',dim,'String',str,'FontSize',12,'FitBoxToText','on');
end