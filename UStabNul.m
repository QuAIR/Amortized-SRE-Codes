function nuu = UStabNul(U)
[rows, cols] = size(U);
n = log2(rows);
PS = PauliString(n);
numCombinations = 4^n;
nu = 0;
for i = 1:numCombinations
    for j = 1:numCombinations
        Pi = PS{i};
        Pj = PS{j};
        nuidx = trace(Pi * U * Pj * U')/2^n;
        if abs(nuidx) > 0.99
            nu = nu + 1;
        end
    end
end
nuu = 2 * n - log2(nu);
end