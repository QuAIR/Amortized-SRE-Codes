% Numerical QFT Choi witnesses using the standard zero-index Fourier matrix.
% Prescribed half-diamond tolerance; no approximate circuit is simulated.
% Certified QFT integers are provided by reproducibility/certify_qft.py.
epsilon = 1e-6;
fprintf('n\tP2(Choi)\tnumerical adaptive T-count witness (epsilon=%g)\n',epsilon);
for n = 3:6
    d = 2^n; indices = 0:d-1;
    U = exp(2*pi*1i*(indices.'*indices)/d)/sqrt(d);
    purity = 2^(-SREChoi(2,U));
    % The 1e-11 guard is numerical, not an interval certificate.
    witness = max(0,ceil(-log2(min(1,purity+16*epsilon))/log2(4/3)-1e-11));
    fprintf('%d\t%.15g\t%d\n',n,purity,witness);
end
