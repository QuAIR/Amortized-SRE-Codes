function [nu, info] = UStabNul(U, mode, tolerance)
%USTABNUL Exact unitary stabilizer nullity, or an explicit floating diagnostic.
%   UStabNul(sym(U)) uses exact symbolic arithmetic (Symbolic Math Toolbox).
%   Construct U exactly, e.g. diag([sym(1), exp(sym(1i)*sym(pi)/4)]).
%   Symbolic conversion cannot recover exact constants lost to rounding.
%   UStabNul(U,'diagnostic',TOL) counts near-unit Pauli-transfer entries.
%   This diagnostic is NOT exact nullity or a certified T bound.
%   No tolerance decides exact Pauli equality for all unitaries.
if nargin < 2, mode = 'exact'; end
mode = validatestring(mode, {'exact','diagnostic'});
[d, columns] = size(U);
if ~ismatrix(U) || d == 0 || d ~= columns || 2^round(log2(d)) ~= d
    error('SRE:Dimension', 'U must be a nonempty square matrix of power-of-two dimension.');
end
n = round(log2(d));
basis = PauliString(n);
count = 0;
if strcmp(mode, 'exact')
    if nargin > 2
        error('SRE:ExactTolerance', 'An exact calculation does not accept a tolerance.');
    end
    if ~isa(U, 'sym')
        error('SRE:ExactInput', 'Exact nullity requires an exactly constructed symbolic matrix. Use diagnostic mode explicitly for floating input.');
    end
    if ~isempty(symvar(U))
        error('SRE:SymbolicParameter', 'Substitute exact parameter values before calculating nullity.');
    end
    if ~all(isAlways(U'*U == sym(eye(d)), 'Unknown','error'), 'all')
        error('SRE:Unitary', 'U must be exactly unitary.');
    end
    for j = 1:numel(basis)
        rotated = U*sym(basis{j})*U';
        for k = 1:numel(basis)
            coefficient = simplify(trace(sym(basis{k})*rotated)/d);
            count = count + double(isAlways(coefficient^2 == 1, 'Unknown','error'));
        end
    end
    info = struct('certified',true, 'mode','exact', 'pauliMappings',count);
else
    if nargin ~= 3
        error('SRE:DiagnosticTolerance', 'Diagnostic mode requires an explicit positive tolerance.');
    end
    validateattributes(tolerance, {'numeric'}, {'scalar','real','finite','positive','<',1});
    ValidateQubitUnitary(U);
    flat = zeros(d*d, numel(basis));
    for j = 1:numel(basis), flat(:,j) = basis{j}(:); end
    for j = 1:numel(basis)
        rotated = U*basis{j}*U';
        coefficients = flat'*rotated(:)/d;
        count = count + nnz(abs(abs(coefficients)-1) <= tolerance);
    end
    warning('SRE:UncertifiedNullity', 'Near-unit PTM counts are floating diagnostics; they cannot certify exact nullity or a T-count lower bound.');
    info = struct('certified',false, 'mode','diagnostic', 'pauliMappings',count, 'tolerance',tolerance);
end
if count == 0
    error('SRE:NoIdentityMapping', 'No identity mapping found; check the input and diagnostic tolerance.');
end
nu = 2*n-log2(count);
end
