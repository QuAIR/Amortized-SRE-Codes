function sre = SRE(alpha, rho)
[rows, cols] = size(rho);
n = log2(rows);
PS = PauliString(n);
num = 4^n;
paulicoesum = 0;
for i=1:num
    paulicoesum = paulicoesum + ((1 / sqrt(2^n) * trace(PS{i} * rho))^2)^alpha;
end
sre = 1/(1-alpha) * log2(paulicoesum) - n;
end

