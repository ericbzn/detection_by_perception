
% detectection_by_perception.m This code uses the human perception principles to detect and recognize landing targets for a UAV application>
% Copyright (C) <2018>  <Eva Dokladalova>
% This code is the Matlab version of Eric Bazan's method for perception principles application to detect and recognize landing targets for a UAV application
% This code does not include the affinity clustering stage

% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.


clear all;
close all;
vis = 1;
scale = 3;
log_size = 7;

% read image
im1 = imread('indir/circle.png');
% rgb -> gl
im2 = rgb2gray(im1);
% add noise
im2n = im2;%imnoise(im2, 'gaussian');
im2n = double(im2n);

% RESIZE IMAGE - PRESERVE CONTOURS
%im2n = imresize (im2n,scale,'cubic');

% MARR-HILDRED
% LOG_GAUSSIAN avec 3 sigma diff/
img = imgaussfilt(im2n,1);
im2nf1 = (del2(img));
im2nf1 = imresize (im2nf1,scale,'bilinear');
img = imgaussfilt(im2n,2);
im2nf2 = (del2(img));
im2nf2 = imresize (im2nf2,scale,'bilinear');
img = imgaussfilt(im2n,3);
im2nf3 = (del2(img));
im2nf3 = imresize (im2nf3,scale);

if vis==1,
figure;imagesc(im2nf1);colormap(gray);colorbar;
figure;imagesc(im2nf2);colormap(gray);colorbar;
figure;imagesc(im2nf3);colormap(gray);colorbar;
end;

% ZERO CROSSING DETECTION
im2nf1 = im2nf1>0;
im2nf2 = im2nf2>0;
im2nf3 = im2nf3>0;

if vis==1,
figure;imagesc(im2nf1);colorbar;
figure;imagesc(im2nf2);colorbar;
figure;imagesc(im2nf3);colorbar;
end;

se = strel('diamond',1);

im2nf1_d = im2nf1 - imerode(im2nf1,se);% - im2nf1;
im2nf2_d = im2nf2 - imerode(im2nf2,se);% - im2nf2;
im2nf3_d = im2nf3 - imerode(im2nf3,se);% - im2nf3;

if vis==1,
figure;imagesc(im2nf1_d);colormap(gray);
figure;imagesc(im2nf2_d);colormap(gray);
figure;imagesc(im2nf3_d);colormap(gray);
end;

% CC LABELING for CONTOURS EXTRACTION
[L1,n1] = bwlabel(im2nf1_d);
[L2,n2] = bwlabel(im2nf2_d);
[L3,n3] = bwlabel(im2nf3_d);

% EXTRACT FEATURES : MEAN GRADIENT INTENSITY AND CIRCULARITY
% GRADIENT IMAGE 
img = imgradient (im2n, 'central');
img = imresize (img, scale, 'bilinear');

if vis==0,
    figure;imagesc(img);colormap(gray);
end;

% create features vector
features = zeros(n1+n2+n3, 2);
[m n] = size(img);
% mean intensity
stats = regionprops(L1,img,'perimeter','Area','ConvexArea','MeanIntensity');
 for i=1:(n1)
     if (stats(i).Area) == 1
         stats(i).Perimeter = 1;
     end;
    features(i,1)= stats(i).MeanIntensity;
    features(i,2)=(4*pi*stats(i).ConvexArea/((stats(i).Perimeter)^2));
end;

stats = regionprops(L2,img,'perimeter','Area','ConvexArea','MeanIntensity');
for i=1:(n2)
    if (stats(i).Area) == 1
         stats(i).Perimeter = 1;
     end;
    features(i+n1,1)= stats(i).MeanIntensity;
    features(i+n1,2)=(4*pi*stats(i).ConvexArea/((stats(i).Perimeter)^2));
end;

stats = regionprops(L3,img,'perimeter','Area','ConvexArea','MeanIntensity');
for i=1:(n3)
    if (stats(i).Area) == 1
         stats(i).Perimeter = 1;
     end;
 features(i+n1+n2,1)= stats(i).MeanIntensity;
 features(i+n1+n2,2)=(4*pi*stats(i).ConvexArea/((stats(i).Perimeter)^2));
end;

rx = hyperRxDetector(features');
xchi2 = chi2pdf(rx,0.9);
ccc = find(xchi2 < 0.000001);

