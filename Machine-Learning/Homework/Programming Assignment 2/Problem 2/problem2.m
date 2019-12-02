clc;clear;
% read the image and view it
img = imread('images/12003.jpg');
subplot(2,3,1); 
imagesc(img); 
axis image;

% extract features (stepsize = 7)
[X, L] = getfeatures(img, 7);
XX = [X(1:2,:) ; X(3:4,:)/10]; % downscale the coordinate features (see part (b))

% kmeans
% Y = kmeans1(XX, 2);

% EM-GMM
% Y = emgmm(XX, 2);

% meanshift
Y = meanshift1(XX);

% make a segmentation image from the labels
segm = labels2segm(Y, L);
subplot(2,3,2); imagesc(segm); axis image;
title('old distance')

% color the segmentation image
csegm = colorsegm(segm, img);
subplot(2,3,3); imagesc(csegm); axis image

% meanshift
Y = meanshift(XX);

% make a segmentation image from the labels
segm = labels2segm(Y, L);
subplot(2,3,5); imagesc(segm); axis image;
title('new distance')

% color the segmentation image
csegm = colorsegm(segm, img);
subplot(2,3,6); imagesc(csegm); axis image

% saveas(gcf,'EM-GMM','svg')
