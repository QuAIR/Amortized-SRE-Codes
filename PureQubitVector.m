function psi = PureQubitVector(state)
%PUREQUBITVECTOR Validate a numerical pure qubit state to tolerance 1e-10.
validateattributes(state, {'numeric'}, {'2d','nonempty','finite'});
tolerance = 1e-10;
if isvector(state)
    psi = state(:);
else
    [d, columns] = size(state);
    if d ~= columns || norm(state-state','fro') > tolerance || abs(trace(state)-1) > tolerance
        error('SRE:PureState', 'The density matrix must be square, Hermitian, trace one and pure.');
    end
    [vectors, values] = eig((state+state')/2, 'vector');
    [~, index] = max(real(values));
    psi = vectors(:,index);
    if norm(state-psi*psi','fro') > tolerance
        error('SRE:PureState', 'Mixed states are unsupported: the Pauli polynomial is not the mixed-state roof.');
    end
end
d = numel(psi);
if 2^round(log2(d)) ~= d
    error('SRE:Dimension', 'State dimension must be a power of two.');
end
if abs(psi'*psi-1) > tolerance
    error('SRE:Normalization', 'The state vector must have unit norm (tolerance 1e-10).');
end
psi = psi/norm(psi);
end
