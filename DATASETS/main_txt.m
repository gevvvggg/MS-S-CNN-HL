clc
clear all
RR=load('dcd2.1.txt');
load T
[height,width]=size(T);
% load classjl
% RR=class;
% [m,n]=size(RR);
% mrclass=[];
% for i=1:m 
%           mrclass=[mrclass RR(i,:)];
% end
mrclass=RR;
%  mrclass=mrclass'+1;
grclass=reshape(mrclass,height,width);
save grclass grclass
% gg=RR';

% load test_label
% r=0;
% for i=1:44742
%     if grclass(i)==test_label(i)
%         r=r+1;
%     end
% end
% f=r/44742