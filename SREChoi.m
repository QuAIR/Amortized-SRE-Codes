function sre = SREChoi(alpha, U)
%SRECHOI Pure-state SRE of the normalized unitary Choi vector, in bits.
%   The Choi witness is a lower bound on amortized SRE, not its general optimum.
d = ValidateQubitUnitary(U);
sre = SRE(alpha, reshape(U.',[],1)/sqrt(d));
end
