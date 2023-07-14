clear all
clc
tic



  load T

 load ALLData;
 Data=ALLData;
 
 load label;
%   train_data =[];
%  train_labels = [];
%  for i = 1 : 3  
%      [a,~] = find(label == i);
%      labels(1:size(a,1),:) = i;
%      train_data = [train_data;Data(label==i,:)];
%      train_labels = [train_labels;labels];
%      clear labels
%  end
%  
%  Train_data  = train_data;
%  Train_labels =  train_labels;
%  clear train_data train_labels
%  [train,test]=crossvalind('holdOut',Train_labels,0.9);
%  train_data=Train_data(train,:);
%  test_data=Train_data(test,:);
%  train_labels=Train_labels(train);
%  test_labels=Train_labels(test);
%  C = cov(train_data);%计算patches的协方差矩阵
%  M = mean(train_data);
%  [V,D] = eig(C);
%  P = V * diag(sqrt(1./(diag(D)))) * V';%P是ZCA Whitening矩阵
%    %对数据矩阵白化前，应保证每一维的均值为0
%  Data= (bsxfun(@minus, Data, M) * P);


a=Data(1:1400*1200,1:6);
c=reshape(a,1400,1200,6);
 b=c;
tt=4;
padn=tt;
aa=padarray(b,[padn,padn],'symmetric');
am=padarray(T,[padn,padn],'symmetric');
 for i=tt+1:1400+tt
    for j=tt+1:1200+tt
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
        
       %  c7=aa((i-tt):(i+tt),(j-tt):(j+tt),7:7);
       % d7=reshape(c7,(tt*2+1)*(tt*2+1),1);
         
       %  c8=aa((i-tt):(i+tt),(j-tt):(j+tt),8:8);
       %  d8=reshape(c8,(tt*2+1)*(tt*2+1),1);
         
       %  c9=aa((i-tt):(i+tt),(j-tt):(j+tt),9:9);
       %  d9=reshape(c9,(tt*2+1)*(tt*2+1),1);
        
        for t1=1:2*tt+1
            for t2=1:2*tt+1
                W(t1,t2)= abs(log(det(c11{5,5})) + trace(c11{5,5} \ c11{t1,t2})+ 30);
            end
        end
       W_d= reshape(W,(tt*2+1)*(tt*2+1),1);    
       for m1=1:81 %121

                    W1(m1,1)=d1(m1,1);
                        W2(m1,1)=d2(m1,1);
                        W3(m1,1)=d3(m1,1);
                        W4(m1,1)=d4(m1,1);
                      W5(m1,1)=d5(m1,1);
                      W6(m1,1)=d6(m1,1);
%                       W7(m1,1)=(W(5,5)/W_d(m1,1))*d1(m1,1);
%                       W8(m1,1)=(W(5,5)/W_d(m1,1))*d2(m1,1);
%                        W9(m1,1)=(W(5,5)/W_d(m1,1))*d3(m1,1);
%                      W10(m1,1)=(W(5,5)/W_d(m1,1))*d4(m1,1);
%                       W11(m1,1)=(W(5,5)/W_d(m1,1))*d5(m1,1);
%                   W12(m1,1)=(W(5,5)/W_d(m1,1))*d6(m1,1);
       end
                R1(i-tt,j-tt,:)=W1;
                     R2(i-tt,j-tt,:)=W2;
                     R3(i-tt,j-tt,:)=W3;
                     R4(i-tt,j-tt,:)=W4;
                      R5(i-tt,j-tt,:)=W5;
                   R6(i-tt,j-tt,:)=W6;
%                      R7(i-tt,j-tt,:)=W7;
%                    R8(i-tt,j-tt,:)=W8;
%                     R9(i-tt,j-tt,:)=W9;
%                    R10(i-tt,j-tt,:)=W10;
%                    R11(i-tt,j-tt,:)=W11;
%               R12(i-tt,j-tt,:)=W12;

    end
 end
             R1=reshape(R1,1400*1200,81);
               R2=reshape(R2,1400*1200,81);
               R3=reshape(R3,1400*1200,81);
               R4=reshape(R4,1400*1200,81);
             R5=reshape(R5,1400*1200,81);
           R6=reshape(R6,1400*1200,81);
%              R7=reshape(R7,750*1024,81);
%             R8=reshape(R8,750*1024,81);
%              R9=reshape(R9,750*1024,81);
%             R10=reshape(R10,750*1024,81);
%             R11=reshape(R11,750*1024,81);
%         R12=reshape(R12,750*1024,81);

%          save R12 R12
%             save R11 R11
%              save R10 R10
%              save R9 R9
%               save R7 R7
%              save R8 R8
            save R6 R6
              save R5 R5
              save R4 R4
              save R3 R3
              save R2 R2
         save R1 R1


toc