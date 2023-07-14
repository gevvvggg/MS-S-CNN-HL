clc
clear all
color = [255,0,0;0,255,0;0,0,255;255,255,0]/255;
% color =[255,0,0;0,0,0;0,255,255;51,102,0;0,0,255;204,102,51;255,255,0;255,0,255;0,255,0;255,255,255]/255;
%RGB = imread('groundtruth.bmp');
%RGB = double(RGB) / 255;
%figure,imshow(RGB)
% n = 1;
load T
[height,width]=size(T);
row = 1200;
col = 1400;
  load label
  load grclass
%    load class
%     label1=reshape(ptest_labels,750,1024);
%                      label1=reshape(final_label,750,1024);
% label1=reshape(gg,750,1024);
%          label1=grclass+1;
%                     label1=final_label;
RGB = zeros(height,width,3);
for ii = 1 : height
    for jj = 1 : width
%            if label(ii,jj)>0
            switch grclass(ii,jj)
                case 1
                    RGB(ii,jj,:) = color(1,:)';
                case 2
                    RGB(ii,jj,:) = color(2,:)';
                case 3
                    RGB(ii,jj,:) = color(3,:)';
                case 4
                    RGB(ii,jj,:) = color(4,:)';
            end
%              else
%                     RGB(ii,jj,:) = color(16,:)' ; 
% % %           
%           end
         end
    end
%              end
figure,imshow(RGB)