function R = RCNN_main(DATA)
%% RCNN's algorithm is used to discriminate
n=size(DATA,1);
    parfor i=1:n
        % data = Fourier_filter(DATA(i,:));

        %% ignition procedure
        data = DATA(i,:);
        ignition=RCNN_exam(data);
        [~,maxposition] = max(data); % Get the peak position
        n0=7;% The pre-peak position
        m0=maxposition-n0;% initial value
        n2=200;% The position behind the peak
        SUM=sum (ignition(m0:maxposition+n2));% Process of integration
        R(i)=SUM;% factor
    end
R=mapminmax(R,0,1);% The normalization process
R=R*200;
% ma = max(R);
% mi = min(R);
% [n,~] = hist(R,ma-mi+1); % Adaptive hist
% % "try" is the error throwing mechanism
%  %% The parameters are obtained by fitting the histogram
% [B,C] = Double_Gaussian_fitting(n,5);
% 
% %% Calculate FOM 
% FOM = (B(2)-B(1))/(1.667*(C(2)+C(1)));
% str_FOM = sprintf('%f',FOM);
% str =['FOM =',str_FOM];
% dim = [.40 .50 .3 .3];
% annotation('textbox',dim,'String',str,'FontSize',12,'FitBoxToText','on');
end