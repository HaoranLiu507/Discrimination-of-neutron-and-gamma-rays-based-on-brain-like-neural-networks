function  D = rand_matrix(d,P,flag,sigma)
% d is dimension; P is probability; flag is matrix type; miu, sigma, and pho are parameters of two dimensional normal distribution
D = ones(d,d);
   if(flag == 'norm') %%
        D = fspecial('gaussian',d,sigma);
        S = 1/D((d+1)/2,(d+1)/2);
        D = D.*S;
        D= rand(d)<(D);
   else
        D = rand(d)<(D.*P);
   end

end