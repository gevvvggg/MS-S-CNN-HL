clc
clear all
load T
%load class
load label

[height,width]=size(T);
m1=0;m2=0;m3=0;m4=0;
%m5=0;m6=0;m7=0;m8=0;m9=0;m10=0;m11=0;m12=0;m13=0;m14=0;m15=0;
in_1=[];in_2=[];in_3=[];in_4=[];
%in_5=[];in_6=[];in_7=[];in_8=[];in_9=[];in_10=[];in_11=[];in_12=[];in_13=[];in_14=[];in_15=[];
              for i=1:height
                  for j=1:width
                      if label(i,j)>0
                          switch label(i,j)
                              case 1
                                  m1=m1+1;
                                  label_1(m1)=label(i,j);
                                  in_1(m1,1)=i;
                                  in_1(m1,2)=j;
                                case 2
                                  m2=m2+1;
                                  label_2(m2)=label(i,j);
                                  in_2(m2,1)=i;
                                  in_2(m2,2)=j;
                                case 3
                                  m3=m3+1;
                                  label_3(m3)=label(i,j);
                                  in_3(m3,1)=i;
                                  in_3(m3,2)=j;
                                case 4
                                  m4=m4+1;
                                  label_4(m4)=label(i,j);
                                  in_4(m4,1)=i;
                                  in_4(m4,2)=j;
%                                   case 5
%                                   m5=m5+1;
%                                   label_5(m5)=label(i,j);
%                                   in_5(m5,1)=i;
%                                   in_5(m5,2)=j;
%                                   case 6
%                                   m6=m6+1;
%                                   label_6(m6)=label(i,j);
%                                   in_6(m6,1)=i;
%                                   in_6(m6,2)=j;
%                                    case 7
%                                   m7=m7+1;
%                                   label_7(m7)=label(i,j);
%                                   in_7(m7,1)=i;
%                                   in_7(m7,2)=j;
%                                   case 8
%                                   m8=m8+1;
%                                   label_8(m8)=label(i,j);
%                                   in_8(m8,1)=i;
%                                   in_8(m8,2)=j;
%                                   case 9
%                                   m9=m9+1;
%                                   label_9(m9)=label(i,j);
%                                   in_9(m9,1)=i;
%                                   in_9(m9,2)=j;
%                                   case 10
%                                   m10=m10+1;
%                                   label_10(m10)=label(i,j);
%                                   in_10(m10,1)=i;
%                                   in_10(m10,2)=j;
%                                   case 11
%                                   m11=m11+1;
%                                   label_11(m11)=label(i,j);
%                                   in_11(m11,1)=i;
%                                   in_11(m11,2)=j;
%                                   case 12
%                                   m12=m12+1;
%                                   label_12(m12)=label(i,j);
%                                   in_12(m12,1)=i;
%                                   in_12(m12,2)=j;
%                                    case 13
%                                   m13=m13+1;
%                                   label_13(m13)=label(i,j);
%                                   in_13(m13,1)=i;
%                                   in_13(m13,2)=j;
%                                   case 14
%                                   m14=m14+1;
%                                   label_14(m14)=label(i,j);
%                                   in_14(m14,1)=i;
%                                   in_14(m14,2)=j;
%                                   case 15
%                                   m15=m15+1;
%                                   label_15(m15)=label(i,j);
%                                   in_15(m15,1)=i;
%                                   in_15(m15,2)=j;
                                  
                          end
                      end
                  end
              end
disp([m1,m2,m3,m4])

Labe_1=randperm(m1);
Labe_2=randperm(m2);
Labe_3=randperm(m3);
Labe_4=randperm(m4);
% Labe_5=randperm(m5);
% Labe_6=randperm(m6);
% Labe_7=randperm(m7);
% Labe_8=randperm(m8);
% Labe_9=randperm(m9);
% Labe_10=randperm(m10);
% Labe_11=randperm(m11);
% Labe_12=randperm(m12);
% Labe_13=randperm(m13);
% Labe_14=randperm(m14);
% Labe_15=randperm(m15);



class=zeros(1400,1200);
M=10000;
%tt=4;

%for i=1:tt%ceil(m1/M)
for i=1:m1
class(in_1(Labe_1(i),1),in_1(Labe_1(i),2))=1;
end
%for i=1:tt%ceil(m2/M)
for i=1:m2
class(in_2(Labe_2(i),1),in_2(Labe_2(i),2))=2;
end
for i=1:m3
class(in_3(Labe_3(i),1),in_3(Labe_3(i),2))=3;
end
for i=1:m4
class(in_4(Labe_4(i),1),in_4(Labe_4(i),2))=4;
end

% for i=1:730%ceil(m5/M)
% class(in_5(Labe_5(i),1),in_5(Labe_5(i),2))=5;
% end
% for i=1:730%ceil(m6/M)
% class(in_6(Labe_6(i),1),in_6(Labe_6(i),2))=6;
% end
% for i=1:730%ceil(m7/M)
% class(in_7(Labe_7(i),1),in_7(Labe_7(i),2))=7;
% end
% for i=1:730%ceil(m8/M)
% class(in_8(Labe_8(i),1),in_8(Labe_8(i),2))=8;
% end
% for i=1:730%fix(m9/M)
% class(in_9(Labe_9(i),1),in_9(Labe_9(i),2))=9;
% end
% for i=1:730%ceil(m10/M)
% class(in_10(Labe_10(i),1),in_10(Labe_10(i),2))=10;
% end
% for i=1:730%ceil(m11/M)
% class(in_11(Labe_11(i),1),in_11(Labe_11(i),2))=11;
% end
% for i=1:730%ceil(m12/M)
% class(in_12(Labe_12(i),1),in_12(Labe_12(i),2))=12;
% end
% for i=1:730%ceil(m13/M)
% class(in_13(Labe_13(i),1),in_13(Labe_13(i),2))=13;
% end
% for i=1:730%ceil(m14/M)
% class(in_14(Labe_14(i),1),in_14(Labe_14(i),2))=14;
% end
% for i=1:730%ceil(m15/M)
% class(in_15(Labe_15(i),1),in_15(Labe_15(i),2))=15;
% end
rgb=zeros(1400,1200,3);
label1=class;
RR=load('f2v7.txt');
grclass=reshape(RR,height,width);
grclass=grclass+1;
% color = [255,0,0;255,128,0;171,138,80;255,255,0;183,0,255;191,191,255;90,11,255;191,255,191;0,252,255;128,0,0;255,182,229;0,255,0;0,131,74;0,0,255;255,217,157]/255;
 color = [255,0,0;0,255,0;0,0,255;255,255,0]/255;
 for i=1:1400
    for j=1:1200
       switch label1(i,j)
             case 1,
                 rgb(i,j,:)=color(1,:)';
             case 2,
                 rgb(i,j,:)=color(2,:)';
             case 3,
                 rgb(i,j,:)=color(3,:)';
             case 4,
                 rgb(i,j,:)=color(4,:)';
           otherwise
                 grclass(i,j)=0;
       end
    end
 end
figure(1),imshow(rgb)
final_label_T=label1;
save final_label_T final_label_T
save grclass grclass

% one_samples_1_2;
% main_mean2;
% select_data_test;
% select_data_train;
