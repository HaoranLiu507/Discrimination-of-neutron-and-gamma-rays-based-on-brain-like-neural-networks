function [data]=guiyihua(data)
n=length(data);
Max=max(data);Min=min(data);
for i=1:n
data(i)=(data(i)-Min)./(Max-Min);
end
