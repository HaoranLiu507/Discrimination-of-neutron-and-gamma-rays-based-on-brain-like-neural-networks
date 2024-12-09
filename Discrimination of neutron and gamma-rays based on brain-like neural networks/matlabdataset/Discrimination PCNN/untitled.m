clc
clear
Pulse_signal = readmatrix('NormalizedDATA.txt');
Num_signal=size(Pulse_signal,1);
R=zeros(1,Num_signal); % R is the discrimination factor
parfor i=1:Num_signal
    Pulse_signal_vector = Pulse_signal(i,:);
    % Ignition_map=PCNN(Pulse_signal_vector);
    %% 
    [m,n] = size(Pulse_signal_vector);
S = im2double(Pulse_signal_vector);
l = 0.1091;
r = 0.1409;
W = [l r l;
     r 0 r;
     l r l];
Y = zeros(m,n); U = Y;TS=Y;
k =180;
M = W;
F = Y; L = Y; U = Y; 
E = Y;
al=0.356;vl=0.0005;ve=15.5;ae=0.081;af=0.325;vf=0.0005;beta=0.67;
for t=1:k
    F = exp(-af) *F +vf*conv2(Y,M,'same') + Pulse_signal_vector;  
	L = exp(-al) *L +vl*conv2(Y,W,'same');              
	U = F.*(1 + beta*L);
    % Y = double(U>E)
    X = 1./(1+exp(E - U));
    Y = double(X > 0.5);
    E = exp(-ae) * E + ve * Y;
    TS =TS+Y;
end
%% 
    [maxvalue,maxposition] = max(Pulse_signal_vector); % Find the maximum position of a pulse signal
    n0=7;
    m0=maxposition-n0;
    n2=123;
    SUM=sum(Ignition_map(m0:maxposition+n2));
    R(i)=SUM;
end
