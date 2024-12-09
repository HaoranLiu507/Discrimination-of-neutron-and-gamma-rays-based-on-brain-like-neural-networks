function R = CC(DATA)
%% The main programme of the Charge comparison method
% Reference: https://doi.org/10.1016/j.radmeas.2010.06.043
% signal = load("nomalizedDATA.txt");
%% Charge comparison calculation
Num_signal=size(DATA,1);
R=zeros(1,Num_signal); % R is the discrimination factor
parfor i=1:Num_signal
    data=DATA(i,:);
    data = Fourier_filter(data);
    [maxvalue,maxposition] = max(data);
    n0=10;
    n1=43;
    n2=300;

    m0=maxposition-n0;
    m1=maxposition+n1;
    m2=maxposition+n2;
    a=maxposition:m1;
    b=m0:m2;
    c0=length(a);
    c1=length(b);
    N=zeros(c0,1);
    M=zeros(c1,1);
    N=data(m1:m2);
    M=data(m0:m2);
    
    sumN=sum(N);
    sumM=sum(M);
    R(i)=sumN/sumM;
end
% %% Figure of Merit Calculation
% R=mapminmax(R,0,1);
% R=R*200;
% Max=max(R);Min=min(R);
% bins = 150;
% [n,~] = hist(R,bins);
% [miu,sigma] = Double_Gaussian_fitting(n,5);
% FOM = (miu(2)-miu(1))/(1.667*(sigma(2)+sigma(1)));
% str_FOM = sprintf('%f',FOM);
% str =['FOM =',str_FOM];
% dim = [.75 .60 .3 .3];
% annotation('textbox',dim,'String',str,'FontSize',12,'FitBoxToText','on');
% line = (miu(1) + 3 *sigma(1) + miu(2) - 3*sigma(2))/2;
% R = mapminmax(R, 0, 150);
% dlmwrite("Charge_comparison_R.txt",R,',')
% Gamma = 0;
% Neutron = 0;
% Error = 0;
% 
% for i = 1:Num_signal
%     if line >= R(i)
%         Gamma = Gamma + 1;
%         Gamma_data(Gamma,:) = signal(i,:);
%         label_C(i) = 1;
%     else
%         Neutron = Neutron + 1;
%         Neutron_data(Neutron,:) = signal(i,:);
%         label_C(i) = 0;
%     end
% 
% %     if label_C(i) ~= labels(i)
% %         Error = Error + 1;
% %     end
% end
% save label_CC.mat label_C
% save Gamma_data.mat Gamma_data
% save Neutron_data.mat Neutron_data
% 1-Error/8404
end
