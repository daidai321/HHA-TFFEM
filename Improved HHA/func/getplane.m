function y = getplane(nn,pcnn, yDirParam)
% function y = getYDir(N, yDirParam)
% Input:
%   nn:           normal field
%   pcnn:         point cloud  
%   yDirParam:    parameters
%                 struct('y0', [0 1 0]', 'angleThresh', [45 15], 'iter', [5 5]);
% Output:
%   y:            Gravity direction

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

	y = yDirParam.y0;
	y = getplaneHelper(nn, pcnn,y, yDirParam.angleThresh(1), yDirParam.iter(1));
    
end

function yDir = getplaneHelper(nn, pcnn, y0, ~, ~)

	yDir = y0;
    
    thresh=[45, 30, 10];
    
    % m
    disthresh=[-0.3,-0.15,-0.05];

	for i = 1:3
		sim0 = yDir'*nn;
		indF = abs(sim0) > cosd(thresh(i));
		indW = abs(sim0) < sind(thresh(i));

		NF = nn(:, indF);
		NW = nn(:, indW);
		A = NW*NW' - NF*NF';
        yDir=getminparam(A,yDir);
        
        sim0 = yDir'*nn;
		indF = abs(sim0) > cosd(thresh(i));
        indW = abs(sim0) < sind(thresh(i));
        
        PP = pcnn(:,indF);
        NP = pcnn(:, indW);
        A=yDir(1);
        B=yDir(2);
        C=yDir(3);
        dtmp=-mean(PP(1,:)*A+PP(2,:)*B+PP(3,:)*C);
        dis=A*pcnn(1,:)+B*pcnn(2,:)+C*pcnn(3,:)+dtmp;
        if i==3
           ind=(dis>disthresh(i))&(dis<-disthresh(2));
        else
           ind=dis>disthresh(i); 
        end
        
        
%         ptCloud = pointCloud(pcnn');
%         cmatrix = ones(size(ptCloud.Location)).*[0.6 0.6 0.6];
%         cmatrix(indF,1)=0.2;
%         cmatrix(indF,2)=0.2;
%         cmatrix(indF,3)=0.85;
%         cmatrix(indW,1)=0.85;
%         cmatrix(indW,2)=0.2;
%         cmatrix(indW,3)=0.2;
%         pcnntmp=pcnn;
%         pcnntmp(:,1)=pcnn(:,3);
%         pcnntmp(:,3)=pcnn(:,1);
%         pcnntmp(:,2)=pcnn(:,2);
%         ptCloud = pointCloud(pcnntmp','Color',cmatrix);
%         pcshow(ptCloud); 
        
        
        nn=nn(:,ind);
        pcnn=pcnn(:,ind);
            
       if i==3
           sim0 = yDir'*nn;
		   indF = abs(sim0) > cosd(thresh(i));
           PF = pcnn(:,indF);
           planes=minsqplane(PF);
           
           yDir=planes';
        end
	end
end

function plane1=minsqplane(ptsc)
   matA=ptsc*ptsc';
   matb=sum(ptsc,2);
   matx=inv(matA)*matb;
   normvalue=1/sqrt(matx'*matx);
   matx=-matx*normvalue;
   plane1=[matx',normvalue];
end

function plane1=svdplane(ptsc)
      meanpt=mean(ptsc,2);
      ptmp = (ptsc - meanpt);
      nmatrix=ptmp*ptmp';
      [U,~,~]=svd(nmatrix);
      plane1=[0,0,0,0];
      plane1(1:3)=U(:,3)';
      plane1(4)=-plane1(1:3)*meanpt;

%     meanpt = np.mean(ptsc, 0)
%     ptmp = (ptsc - meanpt)
%     nmatrix = np.dot(np.transpose(ptmp), ptmp)
%     U, _, _ = np.linalg.svd(nmatrix)
%     plane1 = [0, 0, 0, 0]
%     plane1[:3] = U[:, 2]
%     plane1[3] = -np.dot(plane1[:3], meanpt)
end 


function yDir=getminparam(A,yDir)
    [V, D] = eig(A);
    [~, ind] = min(diag(D));
    newYDir = V(:,ind);
    yDir = newYDir.*sign(yDir'*newYDir);
end
