function [X, L] = getfeatures(img, stepsize)
% getfeatures - extract features from an image
%
%  [X, L] = getfeatures(img, stepsize)
%
% INPUT
%    img      - the image
%    stepsize - window stepsize
%  
% OUTPUT
%    X   - the features: each column is a feature vector
%    L   - structure containing the location information of each feature
%          (used by cluster2segm)

winsize = 7;
if (stepsize > winsize)
  error('stepsize larger than window size');
end

yimg = double(rgb2ycbcr(img));
[sy, sx, sc] = size(img);
offset = floor((winsize-1)/2);

rangex = 1:stepsize:(sx-winsize+1);
rangey = 1:stepsize:(sy-winsize+1);
win = 0:(winsize-1);

X = zeros(4, length(rangex)*length(rangey));

i=1;
for x=rangex
  for y=rangey
    myIu = yimg(y+win, x+win, 2);
    myIv = yimg(y+win, x+win, 3);
    X(:,i) = [mean(myIu(:)) ; ...
              mean(myIv(:)) ; ...
              y + offset ; ...
              x + offset ; ...
              ];
    i = i+1;
  end
end

L.rangex   = rangex;
L.rangey   = rangey;
L.offset   = offset;
L.sx       = sx;
L.sy       = sy;
L.stepsize = stepsize;
L.winsize  = winsize;













