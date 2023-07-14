clear all
%        function ff=confusion(ptest_labels)
load label
%       class=ptestall_labels;
%           class=final_label;
load grclass
             class=grclass;
  %  load class
cluster_num = max(label(:));
num = max(class(:));
a = label(label>0);
b = class(label>0);
confus = zeros(cluster_num,num);
for i = 1 : cluster_num 
    c = find(a == i);
    acc_num(i) = size(c,1);
    for j = 1 : num 
        d = find(b(c) == j);
        confus(i,j) = size(d,1) / size(c,1) * 100;
        confus_num(i,j) = size(d,1);
    end    
end
[s1,s2]=size(confus);
disp(s2)
if s2==3
    b=zeros(4,1);
    confus=[confus,b];
end
confus=confus/100;
% ff=trace(confus)/9;
ff=(confus(1,1)+confus(2,2)+confus(3,3)+confus(4,4))/4;
disp(ff)
m1=0;m2=0;m3=0;m4=0;
load label
for i=1:1400
    for j=1:1200
        switch label(i,j)
            case 1
                 m1=m1+1;
            case 2
                 m2=m2+1;
            case 3
                 m3=m3+1;
            case 4
                 m4=m4+1;
        end
    end
end
m=[m1,m2,m3,m4];
for i=1:4
        Con(i,:)=confus(i,:)*m(i);
end   
for i=1:4
        r(i)=sum(Con(:,i));
end       
for i=1:4
        t(i)=r(i)*m(i);
end 
Pe=sum(t)/sum(m)^2;
Kappa=(ff-Pe)/(1-Pe);
disp(Kappa)