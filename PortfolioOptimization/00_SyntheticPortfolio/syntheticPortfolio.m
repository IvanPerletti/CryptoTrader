clc
clear all
close all
rng(204)
n = 150;
vX = linspace(0,1,n).';
vNoise = rand(n,1);
a1 = -0.2;
b1 = -1.2;
c1 = 0.01;
v = 1000 + 1000* ( vX.^2*a1 + b1*vX + vNoise*c1);
b2 = 0.2;
v2 = 1000 + 1000* ( vX * b2 + vNoise*c1) ;
A = [
    1000 1000 0.0 0.1 0.1;
    1000 1000 -0.3 0.1 0.9;
    1000 1000 -3.0 3.0 0.1;
    1000 1000 0.3 -0.2 0.01;
    1000 1000 -0.3 0.1 0.2;
    ];
mV = zeros(n,size(A,1));
t = datetime(2021,11,11);
formatOut = 'yyyy-mm-dd';

format bank
% [v1 v2 v1*2 v2*2]
fileID = fopen('PortForlioOpt_2021-04-30_2022-02-24.txt','w');
fprintf(fileID,'Date,BTC-EUR,ETH-EUR,EGLD-EUR,CRO-EUR,USD-EUR\r\n');
for ii=1:n
    DateString = datestr(t,formatOut);
    
    fprintf(fileID,'%6s',DateString);
    
    for jj = 1:size(A,1)
        x = vX(ii);
        v = A(jj,:);
        y = v(1) + v(2)*(...
            v(3)*x^2 + v(4)*x + v(5)*rand(1,1));
        fprintf(fileID,',%.2f',y);
        mV(ii,jj) = y;
    end
    fprintf(fileID,'\r\n',t);
    t = t+1;
end
fclose(fileID);

plot(mV)
legend('A','B','C','D','E')