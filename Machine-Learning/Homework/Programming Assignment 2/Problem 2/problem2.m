clc;clear;
% read the image and view it
img = imread('images/12003.jpg');
subplot(1,3,1); 
imagesc(img); 
axis image;

% extract features (stepsize = 7)
[X, L] = getfeatures(img, 7);
XX = [X(1:2,:) ; X(3:4,:)/10]; % downscale the coordinate features (see part (b))

% kmeans
% Y = kmeans(XX, 2);

% EM-GMM
Y = emgmm(XX, 2);

% meanshift
% Y = meanshift(XX);

% make a segmentation image from the labels
segm = labels2segm(Y, L);
subplot(1,3,2); imagesc(segm); axis image;

% color the segmentation image
csegm = colorsegm(segm, img);
















subplot(1,3,3); imagesc(csegm); axis image