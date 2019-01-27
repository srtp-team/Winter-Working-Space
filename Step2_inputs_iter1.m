%%% Constructing PNCC in iteration 1 for CNN inputs %%%
clear;
addpath('../');
%% filelist to train
iter = 1;

root_path = '..\Data\';
data_path = [root_path 'samples\crop\'];
input_path = [root_path 'inputs\iter' num2str(iter) '\'];

filelist = dir([data_path '*.mat']);

filename = filelist(1).name;
load([data_path filename]);
load(['m_iter' num2str(iter) '.mat']);

[std_height,std_width,nChannels1] = size(roi_img);
[std_height,std_width,nChannels2] = size(c3d_map);

ffid = fopen(['../bin/imgList_iter' num2str(iter) '.txt'], 'wt');

for fi = 1:length(filelist)

    filename = filelist(fi).name;
    load([data_path filename]);
    
    map = zeros(std_height, std_width, nChannels1+nChannels2);
    
    for j = 1:nChannels1
        map(:,:,j) = roi_img(:,:,j)/255;
    end
    
    for j = nChannels1+1:nChannels1+nChannels2
        map(:,:,j) = c3d_map(:,:,j-nChannels1);
    end
    
    pad = zeros(std_height, std_width, 2);
    
    map = [map(:)-m; pad(:)];
    
    fid = fopen([input_path filename(1:end-4) '.input'], 'wb');
    fwrite(fid, length(map(:)), 'int32');
    fwrite(fid, map(:),  'single');
    fclose(fid);
    
    para0 = para0(:);
    save([input_path filename(1:end-4) '.mat'], 'para0');
    
    
    fullname = [input_path filename(1:end-4) '.input'];
    fullname = strrep(fullname, '\', '/');
    fprintf(ffid, [fullname, '\n']);

    
end

fclose(ffid);
