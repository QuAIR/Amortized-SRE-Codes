function basis = PauliString(n)
%PAULISTRING Hermitian, unsigned Pauli basis in lexicographic I,X,Y,Z order.
%   Stores 4^n dense 2^n by 2^n matrices; use only for small n.
validateattributes(n, {'numeric'}, {'scalar','real','finite','integer','nonnegative'});
one = {eye(2), [0 1;1 0], [0 -1i;1i 0], diag([1 -1])};
basis = {1};
for site = 1:n
    next = cell(1,4*numel(basis));
    for j = 1:numel(basis)
        for k = 1:4, next{4*(j-1)+k} = kron(basis{j},one{k}); end
    end
    basis = next;
end
end
