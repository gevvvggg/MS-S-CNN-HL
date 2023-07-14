clear all
clc
tic



 load T
 T=T;
 load ALLData;
 Data=ALLData;
 load label;
 load final_label_T
%     load final_label
% load final_label_1
%    final_label_T=final_label;
% final_label_T=label;
 train_data =[];
train_labels = [];
for i = 1 : 15
    [a,~] = find(final_label_T == i);
    labels(1:size(a,1),:) = i;
    train_data = [train_data;Data(final_label_T==i,:)];
    train_labels = [train_labels;labels];
    clear labels
end


Train_data  = train_data;
Train_labels =  train_labels;
clear train_data train_labels
[train,test]=crossvalind('holdOut',Train_labels,0.9);
train_data=Train_data(train,:);
test_data=Train_data(test,:);
train_labels=Train_labels(train);
test_labels=Train_labels(test);
C = cov(train_data);%计算patches的协方差矩阵
M = mean(train_data);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D)))) * V';%P是ZCA Whitening矩阵
  %对数据矩阵白化前，应保证每一维的均值为0
Data= (bsxfun(@minus, Data, M) * P);

a=Data(1:1400*1200,1:6);
b=reshape(a,1400,1200,6);
tt=4;%5
padn=tt;
aa=padarray(b,[padn,padn],'symmetric');
am=padarray(T,[padn,padn],'symmetric');
ff=1;
 for i=tt+1:1400+tt
    for j=tt+1:1200+tt
        if (final_label_T(i-tt,j-tt)~=0)
        c1=aa((i-tt):(i+tt),(j-tt):(j+tt),1:1);
        c11=am((i-tt):(i+tt),(j-tt):(j+tt),1:1);
        d1=reshape(c1,(tt*2+1)*(tt*2+1),1);
        
        c2=aa((i-tt):(i+tt),(j-tt):(j+tt),2:2);
        d2=reshape(c2,(tt*2+1)*(tt*2+1),1);
        
        c3=aa((i-tt):(i+tt),(j-tt):(j+tt),3:3);
        d3=reshape(c3,(tt*2+1)*(tt*2+1),1);
     
        c4=aa((i-tt):(i+tt),(j-tt):(j+tt),4:4);
        d4=reshape(c4,(tt*2+1)*(tt*2+1),1);
        
        c5=aa((i-tt):(i+tt),(j-tt):(j+tt),5:5);
        d5=reshape(c5,(tt*2+1)*(tt*2+1),1);
     
        c6=aa((i-tt):(i+tt),(j-tt):(j+tt),6:6);
        d6=reshape(c6,(tt*2+1)*(tt*2+1),1);
        
%         c7=aa((i-tt):(i+tt),(j-tt):(j+tt),7:7);
%         d7=reshape(c7,(tt*2+1)*(tt*2+1),1);
%         
%         c8=aa((i-tt):(i+tt),(j-tt):(j+tt),8:8);
%         d8=reshape(c8,(tt*2+1)*(tt*2+1),1);
%         
%         c9=aa((i-tt):(i+tt),(j-tt):(j+tt),9:9);
%         d9=reshape(c9,(tt*2+1)*(tt*2+1),1);
%         
%计算c11这个5*5的矩阵的行列式的值加上c11除以c11中的每一个元素加上30，最后去绝对值。W(1,1)~W(9,9)共有81个值
%将W的值赋给W_d，大小为(81,1)
        for t1=1:9
            for t2=1:9
                W(t1,t2)= abs(log(det(c11{5,5})) + trace(c11{5,5} \ c11{t1,t2})+ 30);
            end
        end
       W_d= reshape(W,(tt*2+1)*(tt*2+1),1);
%
       for m1=1:81

           W1(m1,1)=d1(m1,1);
           W2(m1,1)=d2(m1,1);
           W3(m1,1)=d3(m1,1);
           W4(m1,1)=d4(m1,1);
           W5(m1,1)=d5(m1,1);
           W6(m1,1)=d6(m1,1);
%            W7(m1,1)=((W(5,5)/W_d(m1,1)))*d1(m1,1);
%            W8(m1,1)=((W(5,5)/W_d(m1,1)))*d2(m1,1);
%            W9(m1,1)=((W(5,5)/W_d(m1,1)))*d3(m1,1);
%            W10(m1,1)=((W(5,5)/W_d(m1,1)))*d4(m1,1);
%            W11(m1,1)=((W(5,5)/W_d(m1,1)))*d5(m1,1);
%            W12(m1,1)=((W(5,5)/W_d(m1,1)))*d6(m1,1);
       end
       R1(ff,:)=real(W1);
       R2(ff,:)=real(W2);
       R3(ff,:)=real(W3);
       R4(ff,:)=real(W4);
       R5(ff,:)=real(W5);
       R6(ff,:)=real(W6);
%       R7(ff,:)=real(W7);
%       R8(ff,:)=real(W8);
%       R9(ff,:)=real(W9);
%       R10(ff,:)=real(W10);
%       R11(ff,:)=real(W11);
%       R12(ff,:)=real(W12);
       r_label(ff)=final_label_T(i-tt,j-tt);
     

ff=ff+1;
        end
    end
 end

 

 save R1 R1 
 save R2 R2
 save R3 R3
 save R4 R4
 save R5 R5
 save R6 R6
% save R7 R7
% save R8 R8
% save R9 R9
% save R10 R10
% save R11 R11
% save R12 R12
 save r_label r_label

toc