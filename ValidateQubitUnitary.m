function d = ValidateQubitUnitary(U)
%VALIDATEQUBITUNITARY Numerical validation, not an exact-arithmetic certificate.
validateattributes(U, {'numeric'}, {'2d','nonempty','finite'});
[d, columns] = size(U);
if d ~= columns || 2^round(log2(d)) ~= d
    error('SRE:Dimension', 'U must be square with power-of-two dimension.');
end
if max(abs(U'*U-eye(d)), [], 'all') > 1e-10
    error('SRE:Unitary', 'U must be unitary within absolute tolerance 1e-10.');
end
end
