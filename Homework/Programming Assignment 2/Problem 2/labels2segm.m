function segm = labels2segm(Y, L)
% labels2segm - form a segmentation image using cluster labels
%
%   segm = labels2segm(Y, L)
%
%  Y - cluster labels for each location
%  L - location structure from getfeatures
%
%  segm - output segmentation image

segm = zeros(L.sy, L.sx);
  
rstep = floor(L.stepsize/2);

stepbox = [-rstep:(L.stepsize-1-rstep)];

rx = L.rangex + L.offset;
ry = L.rangey + L.offset;

for i=stepbox
  for j=stepbox
    segm(ry+j, rx+i) = reshape(Y, length(ry), length(rx));
  end
end

%%% now fill in the borders if they are missing %%%
minx = min(rx) + stepbox(1) - 1;
maxx = max(rx) + stepbox(end) + 1;
miny = min(ry) + stepbox(1) - 1;
maxy = max(ry) + stepbox(end) + 1;
  
if (1 <= minx)
  % fill in left edge
  for xx = 1:minx
    segm(:,xx,:) = segm(:,minx+1,:);
  end
end
if (maxx <= L.sx)
  % fill in right edge
  for xx = maxx:L.sx
    segm(:,xx,:) = segm(:,maxx-1,:);
  end
end
if (1 <= miny)
  % fill in top edge
  for yy=1:miny
    segm(yy,:,:) = segm(miny+1,:,:);
  end
end
if (maxy <= L.sy)
  % fill in bottom edge
  for yy = maxy:L.sy
    segm(yy,:,:) = segm(maxy-1,:,:);
  end
end

