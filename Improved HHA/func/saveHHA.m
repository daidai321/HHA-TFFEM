function [HHA,plane] = saveHHA(imName, C, outDir, D, RD)
% function HHA = saveHHA(imName, C, outDir, D, RD)

% AUTORIGHTS

  if(isempty(D)), D = getImage(imName, 'depth'); end
  if(isempty(RD)), RD = getImage(imName, 'rawdepth'); end
  
  RD = double(RD)./1000;  missingMask = RD == 0;
  [pc, N, plane] = processDepthImage(RD, missingMask, C);
  
% H  
  Hd=RD/max(RD(:));
  % use DEM
  % labeldistrube=load('labeld_kitti.txt');
  % Hd=DEM(RD,labeldistrube);
  Hd(RD==0)=0;
  
% H
  height=pc(:,:,1)*plane(1)+pc(:,:,2)*plane(2)+pc(:,:,3)*plane(3)+plane(4);
  height(height<0)=0;
  height(height>2.55)=2.55;
  height=height/2.55;
  height(RD==0)=0;
  
% A
  [h,w,~]=size(N);
  Nr=reshape(N,[h*w,3]);
  angl = -Nr* plane(1:3);
  angl=reshape(angl,[h,w]);
  angl=acos(angl)/pi;
  angl(RD==0)=0;
  
  % HHA
  I(:,:,1) = Hd; 
  I(:,:,2) = height;
  I(:,:,3) = angl; 
  I = uint8(I*255);
  HHA = I;
  
  imwrite(HHA,'test.png');
  imshow(HHA);
  
  % Save if can save
  if(~isempty(outDir) && ~isempty(imName)), imwrite(I, fullfile_ext(outDir, imName, 'png')); end

end
