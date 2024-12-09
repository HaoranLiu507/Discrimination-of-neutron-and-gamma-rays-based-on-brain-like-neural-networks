function R = SDCC(DATA,PARAM)
R = zeros(size(DATA,1),1);
DATA = filter_my(DATA,PARAM);

parfor i = 1:size(DATA,1)
    data = DATA(i,:)
    [max_value, max_position] = max(data);
    ROI = data(max_position+20:end);
    R(i) = log(sum(ROI.^2));
end
end
