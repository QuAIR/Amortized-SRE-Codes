clear;

QFT3 = zeros(8,8);
for i=1:8
for k=1:8
    QFT3(i,k) = 1/sqrt(8) * exp(2*pi*1j*i*k/8);
end
end
QFT4 = zeros(16,16);
for i=1:16
for k=1:16
    QFT4(i,k) = 1/sqrt(16) * exp(2*pi*1j*i*k/16);
end
end
QFT5 = zeros(32,32);
for i=1:32
for k=1:32
    QFT5(i,k) = 1/sqrt(32) * exp(2*pi*1j*i*k/32);
end
end
QFT6 = zeros(64,64);
for i=1:64
for k=1:64
    QFT6(i,k) = 1/sqrt(64) * exp(2*pi*1j*i*k/64);
end
end

disp("========== unitary Tcount ===========")

%%
UStabNul(QFT6)
SREChoi(2,QFT6)/(2-log2(3))