classdef TestSRE < matlab.unittest.TestCase
    methods (Test)
        function knownValues(test)
            t = diag([1 exp(1i*pi/4)]);
            ccz = diag([ones(1,7) -1]);
            test.verifyEqual(SRE(2,t*ones(2,1)/sqrt(2)),log2(4/3),'AbsTol',1e-12);
            test.verifyEqual(SRE(1,t*ones(2,1)/sqrt(2)),.5,'AbsTol',1e-12);
            test.verifyEqual(SREChoi(2,t),log2(4/3),'AbsTol',1e-12);
            test.verifyEqual(SREChoi(2,ccz),log2(32/11),'AbsTol',1e-12);
            bell = [1;0;0;1]/sqrt(2);
            test.verifyEqual(SRE(2,bell*bell'),0,'AbsTol',1e-12);
        end
        function directPauliCrossCheck(test)
            rng(1743);
            for n = 1:4
                psi = randn(2^n,1)+1i*randn(2^n,1); psi = psi/norm(psi);
                basis = PauliString(n);
                for alpha = [1.5 2 3]
                    total = 0;
                    for j = 1:numel(basis), total = total+abs(psi'*basis{j}*psi)^(2*alpha)/2^n; end
                    test.verifyEqual(PauliMoments(psi,alpha,3),total,'AbsTol',2e-12);
                end
            end
        end
        function renyiContinuityAtOne(test)
            psi = [1;exp(1i*pi/4)]/sqrt(2);
            for alpha = [1-1e-8 1+1e-8 1-1e-12 1+1e-12 1+eps]
                expected = log1p(.5*expm1(-(alpha-1)*log(2)))/((1-alpha)*log(2));
                test.verifyEqual(SRE(alpha,psi),expected,'AbsTol',2e-13);
                test.verifyEqual(SRE(alpha,kron(kron(psi,psi),psi)),3*expected,'AbsTol',2e-12);
            end
        end
        function invalidInputs(test)
            test.verifyError(@() SRE(2,eye(2)/2),'SRE:PureState');
            test.verifyError(@() SRE(2,[1;1]),'SRE:Normalization');
            test.verifyError(@() SRE(2,[1;0;0]),'SRE:Dimension');
            test.verifyError(@() SREChoi(2,2*eye(2)),'SRE:Unitary');
            test.verifyError(@() SREChoi(2,eye(3)),'SRE:Dimension');
            test.verifyError(@() UStabNul(eye(2)),'SRE:ExactInput');
            test.verifyError(@() UStabNul(eye(2),'diagnostic'),'SRE:DiagnosticTolerance');
        end
        function qftConvention(test)
            j = 0:3; u = exp(2*pi*1i*(j.'*j)/4)/2;
            test.verifyEqual(2^(-SREChoi(2,u)),7/16,'AbsTol',1e-12);
        end
        function exactSymbolicNullity(test)
            t = diag([sym(1), exp(sym(1i)*sym(pi)/4)]);
            [nu,info] = UStabNul(t);
            test.verifyEqual(nu,1); test.verifyTrue(info.certified);
            test.verifyEqual(UStabNul(sym([1 1;1 -1])/sqrt(sym(2))),0);
            test.verifyEqual(UStabNul(diag(sym([ones(1,7) -1]))),3);
            theta = sym(pi)/100;
            u = diag([exp(-sym(1i)*theta/2),exp(sym(1i)*theta/2)]);
            test.verifyEqual(UStabNul(u),1);
            syms x real
            test.verifyError(@() UStabNul(diag([sym(1) exp(sym(1i)*x)])), 'SRE:SymbolicParameter');
        end
        function legacyThresholdCounterexample(test)
            theta = pi/100;
            u = diag([exp(-1i*theta/2) exp(1i*theta/2)]);
            test.verifyWarning(@() UStabNul(u,'diagnostic',.01),'SRE:UncertifiedNullity');
            prior = warning('off','SRE:UncertifiedNullity');
            cleanup = onCleanup(@() warning(prior)); %#ok<NASGU>
            [nu,info] = UStabNul(u,'diagnostic',.01);
            test.verifyEqual(nu,0); test.verifyFalse(info.certified);
            test.verifyEqual(UStabNul(u,'diagnostic',1e-12),1);
        end
    end
end
