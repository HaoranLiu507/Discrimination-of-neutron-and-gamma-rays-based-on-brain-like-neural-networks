function TS = CCNN_exam(I)
    S = double(I);
    [m, n] = size(I);
    TS = 0;
    Y = zeros(m, n);
    E = Y + 1;
    U = Y;
    F = Y;
    L = Y;
    % af = 0.58;
    % ae = 0.59;
    % al = 0.15;
    % Ve = 6.0;
    % beta = 0.77;
    % Vf = 0.63;
    % Vl = Vf;
    % CCNN_K = 100;
    af = 0.15;
    ae = 0.69;
    al = 0.25;
    Ve = 8.0;
    beta = 0.55;
    Vf = 0.53;
    Vl = Vf;
    CCNN_K = 100;

    M = [0.5 1 0.5; 1 0 1; 0.5 1 0.5];  
    W = M;

    for t = 1:CCNN_K-1  
        F = exp(-af) .* F + Vf * conv2(Y, M, 'same') + S;  
        L = exp(-al) .* L + Vl * conv2(Y, W, 'same');
        U = F .* (1 + beta * L);
        Y = 1 ./ (1 + exp(E - U));
        E = exp(-ae) .* E + Ve * Y;
        TS = TS + Y;
    end
end

