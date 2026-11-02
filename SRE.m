function sre = SRE(alpha, state)
%SRE Stabilizer Renyi entropy in bits, for a pure qubit state.
%   SRE(ALPHA, PSI) accepts a normalized vector or pure density matrix.
%   ALPHA must be positive; ALPHA=1 uses the Shannon limit.
validateattributes(alpha, {'numeric'}, {'scalar','real','finite','positive'});
psi = PureQubitVector(state);
if alpha == 1
    [~, sre] = PauliMoments(psi, 1);
elseif abs(alpha-1) < 1e-3
    [~, ~, sre] = PauliMoments(psi, alpha);
else
    moment = PauliMoments(psi, alpha);
    sre = log2(moment)/(1-alpha);
end
end

