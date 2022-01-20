function [pc, N, plane] = processDepthImage(z, missingMask, C)

% AUTORIGHTS

  yDirParam.angleThresh = [45 15];
  yDirParam.iter = [5 5];
  yDirParam.y0 = [0 1 0]';
  normalParam.patchSize = [3 10];
  [X, Y, Z] = getPointCloudFromZ(z, C, 1);
  pc = cat(3, X, Y, Z);
  
  % Compute the normals for this image
  [N1 , ~] = computeNormalsSquareSupport(z, missingMask, normalParam.patchSize(1),...
    1, C, ones(size(z)));

   nn = permute(N1,[3 1 2]);     
   pcnn=permute(pc,[3 1 2]);
   nn = reshape(nn,[3 numel(nn)/3]);
   pcnn=reshape(pcnn,[3 numel(pcnn)/3]);
   
   %%%%KITTI%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Keep point clouds and normals within a certain range (only for kitti
   % dataset)
   [~,w,~]=size(N1);
   if w>1000
       nn = nn(:,pcnn(3,:)>0);  
       pcnn = pcnn(:,pcnn(3,:)>0);  
       nn = nn(:,pcnn(3,:)<50);  
       pcnn = pcnn(:,pcnn(3,:)<50);

       nn = nn(:,pcnn(1,:)>-10);  
       pcnn = pcnn(:,pcnn(1,:)>-10);  
       nn = nn(:,pcnn(1,:)<10);  
       pcnn = pcnn(:,pcnn(1,:)<10);  

       nn = nn(:,pcnn(2,:)>-1);  
       pcnn = pcnn(:,pcnn(2,:)>-1);  
   end
   % pcshow(pcnn');

   N = N1; 
   % Compute the direction of gravity
   plane=getplane(nn,pcnn,yDirParam);


  
  
end
