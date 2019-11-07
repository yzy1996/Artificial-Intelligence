function csegm = colorsegm(segm, img)
% colorsegm - color a segmentation based on the image
%
%   csegm = colorsegm(segm, img)
% 
%  segm = the segmentation image
%   img = the original image
%
% csegm = the colored segmentation -- each segment is colored based on the 
%         average pixel color within the segment.

rimg = img(:,:,1);
gimg = img(:,:,2);
bimg = img(:,:,3);
for j=1:max(segm(:))
  ii = find(segm==j);
  rimg(ii) = mean(rimg(ii));
  gimg(ii) = mean(gimg(ii));
  bimg(ii) = mean(bimg(ii));
end
csegm = cat(3, rimg, gimg, bimg);
