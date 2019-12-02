clc;clear;

imgfile = dir('images/*.jpg');
for i = 1:1
    img = imread(['images/' imgfile(i).name]);
    figure
    subplot(3,3,4);
    title('Original image')
    imagesc(img);
    axis image;
    
    % extract features (stepsize = 7)
    [X, L] = getfeatures(img, 7);
    XX = [X(1:2,:) ; X(3:4,:)/10]; % downscale the coordinate features (see part (b))

%%%%%%%%%%%%%%%%%%%%%%%%%%  
    % kmeans
    Y = kmeans(XX, 2);
    
    % make a segmentation image from the labels
    segm = labels2segm(Y, L);
    subplot(3,3,2); imagesc(segm); axis image;
    title('K-means')
    
    % color the segmentation image
    csegm = colorsegm(segm, img);
    subplot(3,3,3); imagesc(csegm); axis image
    
%%%%%%%%%%%%%%%%%%%%%%%%%%     
    % EM-GMM
    Y = emgmm(XX, 2);
    % make a segmentation image from the labels
    segm = labels2segm(Y, L);
    subplot(3,3,5); imagesc(segm); axis image;
    title('EM-GMM')
    
    % color the segmentation image
    csegm = colorsegm(segm, img);
    subplot(3,3,6); imagesc(csegm); axis image
    
%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % meanshift
    Y = meanshift(XX);
    % make a segmentation image from the labels
    segm = labels2segm(Y, L);
    subplot(3,3,8); imagesc(segm); axis image;
    title('Mean-shift')
    
    % color the segmentation image
    csegm = colorsegm(segm, img);
    subplot(3,3,9); imagesc(csegm); axis image
   
    saveas(gcf, ['fig' imgfile(i).name(1:end-4)], 'svg')
end
