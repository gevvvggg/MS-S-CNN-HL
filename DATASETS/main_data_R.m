clc
clear all
load R6
load R5
load R3
load R4
load R2
load R1
%  load R7
%  load R8
%  load R9
%  load R10
%  load R11
%  load R12
%  Rw=R6;
load label;
%  load final_label_T
%  m=1;
%  n=1;
%  for i=1:1400*1200
%       test_data(n,:,:)=Rw(i,:,:);
%          test_label(n)=1;
%          n=n+1;
%      if final_label_T(i)>0
%          train_data(m,:)=Rw(i,:);
%          train_label(m)=final_label_T(i);
%          m=m+1;
%      end
%  end
%  
 
 
 test_data6=R6;
 test_data5=R5;
 test_data4=R4;
 test_data3=R3;
 test_data2=R2;
 test_data1=R1;
%  test_data7=R7;
%  test_data8=R8;
%  test_data9=R9;
%  test_data10=R10;
%  test_data11=R11;
%  test_data12=R12; 

%  train_data6=train_data;

save test_data6 test_data6
save test_data5 test_data5
save test_data4 test_data4
save test_data3 test_data3
save test_data2 test_data2
save test_data1 test_data1
%  save test_data7 test_data7
%  save test_data8 test_data8
%  save test_data9 test_data9
%  save test_data10 test_data10
%  save test_data11 test_data11
%  save test_data12 test_data12

