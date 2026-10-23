function [moments, shannonSRE, renyiSRE] = PauliMoments(psi, orders, batch)
%PAULIMOMENTS P_alpha=2^-n sum_P |<psi|P|psi>|^(2 alpha).
%   XOR diagonals and Walsh transforms avoid storing the dense Pauli basis.
%   Work O(n 4^n), memory O(batch 2^n). This is an exponential classical
%   calculation, not an efficient experimental measurement protocol.
%   Third output gives entropies with a stable limit near alpha=1.
if nargin < 3, batch = 32; end
psi = PureQubitVector(psi);
validateattributes(orders, {'numeric'}, {'vector','nonempty','real','finite','positive'});
validateattributes(batch, {'numeric'}, {'scalar','real','finite','integer','positive'});
d = numel(psi); indices = uint64(0:d-1);
moments = zeros(size(orders)); entropy = 0; deltaMoments = zeros(size(orders));
for start = 0:batch:d-1
    shifts = uint64((start:min(start+batch-1,d-1))');
    permutation = double(bitxor(shifts,indices))+1;
    a = psi.' .* conj(reshape(psi(permutation),size(permutation)));
    stride = 1;
    while stride < d
        for first = 1:2*stride:d
            left = a(:,first:first+stride-1);
            right = a(:,first+stride:first+2*stride-1);
            a(:,first:first+stride-1) = left+right;
            a(:,first+stride:first+2*stride-1) = left-right;
        end
        stride = 2*stride;
    end
    squares = abs(a).^2;
    for k = 1:numel(orders)
        moments(k) = moments(k)+sum(squares.^orders(k),'all')/d;
        if nargout > 2 && orders(k) ~= 1 && abs(orders(k)-1) < 1e-3
            positive = squares(squares > 0);
            deltaMoments(k) = deltaMoments(k)+sum(positive.*expm1((orders(k)-1)*log(positive)))/d;
        end
    end
    if nargout > 1
        probabilities = squares(squares > 0)/d;
        entropy = entropy-sum(probabilities.*log2(probabilities));
    end
end
shannonSRE = entropy-log2(d);
if nargout > 2
    renyiSRE = zeros(size(orders));
    for k = 1:numel(orders)
        if orders(k) == 1
            renyiSRE(k) = shannonSRE;
        elseif abs(orders(k)-1) < 1e-3
            % For a normalized pure state P_1=1 exactly; compute P_alpha-1
            % directly instead of subtracting nearly equal floating numbers.
            renyiSRE(k) = log1p(deltaMoments(k))/((1-orders(k))*log(2));
        else
            renyiSRE(k) = log2(moments(k))/(1-orders(k));
        end
    end
end
end
