clc
clear
addpath(genpath(pwd));

%% epflcord
% SamplePath1 =  'D:/dataset/RGB-D/EPFL/epfl_corridor/20141008_141749_00/'; 
% 
% savedir= './epfl_corridor/';
% C=[365.8046307762528, 0.0, 254.31510758228475
%    0.0, 365.80463336811664, 206.98513348550657
%    0.0, 0.0, 1.0];
% plane = [0.05856724728746367, 0.9678837910705262, 0.2444807651495611, -1.697096896703962];


%% epfllab
% SamplePath1 =  'D:/dataset/RGB-D/EPFL/epfl_lab/20140804_160621_00/'; 
% savedir= './epfl_lab/';
% C=[365.8046307762528, 0.0, 254.31510758228475
%    0.0, 365.80463336811664, 206.98513348550657
%    0.0, 0.0, 1.0];
% plane = [0.0037380404447673473, 0.9963206002336535, 0.08562294437640966, -1.955931620009659];

%% unihall
% SamplePath1 =  'D:/dataset/RGB-D/unihall/dataeval/depth01/'; 
% savedir= './unihall/';
% C=[591.04053696870778, 0.0, 242.73913761751615
%    0.0, 594.21434211923247, 300.6922
%    0.0, 0.0, 1.0];
% plane = [0.05406682176524861, 0.997783198617921, -0.03880035231793, -1.0430059738651149];

%% ktp
% SamplePath1 =  'D:/dataset/RGB-D/KTP_dataset_images/images/stilldepth/'; 
% savedir= './ktp/';
% C=[525.0, 0.0, 319.5
%    0.0, 525.0, 239.5
%    0.0, 0.0, 1.0];
% plane = [-0.003444, 0.996118, 0.087957, -1.305195];

%% kitti
SamplePath1 =  'D:/dataset/RGB-D/kitti_out/depth/'; 
camerapath1='D:/dataset/RGB-D/kitti_out/camera/'; 
savedir= './kitti/';


%%
fileExt = '*.png'; 
files = dir(fullfile(SamplePath1,fileExt)); 
len1 = size(files,1);
len1=fix(len1);

knum=0;
fid = fopen('ydir.txt', 'w');

for i=1:len1
    fileName = strcat(SamplePath1,files(i).name); 
    savepath = strcat(savedir,files(i).name);
    RD = imread(fileName);
    RD(RD>8000)=0;
    
    % The KITTI needs to read camera parameters for each image individually while other datasets have fixed camera parameters.
    if contains(SamplePath1,'kitti')
        camerapath=strcat(camerapath1,files(i).name);
        camerapath=camerapath(1:end-4);
        camerapath=strcat(camerapath,'.txt');
        C=load(camerapath);
        C=C(1:3,1:3);
        
        % Unify depth units to mm
        RD=RD*10;
    end
    
    D=RD;
    
    [HHA,plane] = saveHHA([], C, [], D, RD); 

%   imwrite(I, savepath);
    fprintf(fid,'%s\n',files(i).name);
    fprintf(fid,'%f\t%f\t%f\t%f\n',plane(1),plane(2),plane(3),plane(4));
 
end
fclose(fid);












