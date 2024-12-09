function R = FGA(DATA,PARAM)
R = zeros(1,size(DATA,1));
point = 80;
n = 1:1:length(DATA(1,point:end));
N = length(DATA(1,point:end));
DATA = filter_my(DATA(:,point:end),PARAM);
for i = 1:size(DATA,1)
    data = DATA(i,:);
    X_0 = abs(sum(data));
    X_1 = abs(sqrt(power((sum(data.*cos(2*pi.*n/N))),2)+power((sum(data.*sin(2*pi.*n/N))),2)));
    R(i) = abs(X_0 - X_1);
end
end