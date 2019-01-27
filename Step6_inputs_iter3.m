%%% Constructing PNCC in iteration 3 for CNN inputs %%%

clear;

iter = 3; % set the iteration

root_path = '..\Data\';
src_path = [root_path 'inputs\iter' num2str(iter-1) '\'];
des_path = [root_path 'inputs\iter' num2str(iter) '\'];
img_path = [root_path 'samples\crop\'];

std_size = 100;
width = std_size;
height = std_size;
nChannels1 = 3;
nChannels2 = 3;

pca_co = 40;
pca_exp_co = 13;
para_nums = 234;


load('Model_Expression.mat');
load('Model_Shape.mat');
load('params_attr.mat');
load('vertex_code.mat');
load(['m_iter' num2str(iter) '.mat']);

mu = mu_shape + mu_exp;

count = 0;

filelist = dir([src_path '*.input']);

ffid = fopen(['../bin/imgList_iter' num2str(iter) '.txt'], 'wt');

for fi = 1:length(filelist);
    filename = filelist(fi).name(1:end-6);
    
    % load img
    load([img_path filename '.mat']);
    roi_img = roi_img / 255;
    
    % load parameters
    load([src_path filename '.mat']);
      
    % load para_delta
    fid = fopen([src_path filename '.input.dat']);
    feat = fread(fid, para_nums, 'single');
    fclose(fid);
    
    
    %% 1. The current para0
    % load the regressed para_delta
    feat = [feat(1:5,:); zeros(1, size(feat,2)); feat(6:end,:)];

    std_mat = repmat(params_std, 1, size(feat,2));
    feat = feat .* std_mat;  
    shape_ind = 7+1:7+199; shape_void = shape_ind(pca_co+1:end);
    exp_ind = 7+199+1:7+199+29; exp_void = exp_ind(pca_exp_co+1:end);
    feat([shape_void, exp_void],:) = 0;
    para0 = para0 + feat;
        
    %% 2. The PNCC
    Pose_Para0 = para0(1:7);
    Shape_Para0 = para0(7+1:7+199,:);
    Exp_Para0 = para0(7+199+1:end, :);
    c3d_map = CodeMap(mu, w, w_exp, tri, vertex_code, roi_img, Pose_Para0, Shape_Para0, Exp_Para0);

    map = zeros(height, width, nChannels1+nChannels2);

    for j = 1:nChannels1
        map(:,:,j) = roi_img(:,:,j);
    end

    for j = nChannels1+1:nChannels1+nChannels2
        map(:,:,j) = c3d_map(:,:,j-nChannels1);
    end    

    pad = zeros(height, width, 2);
    
    map = [map(:)-m; pad(:)];
    
    %% save
    fid = fopen([des_path filename '.input'], 'wb');
    fwrite(fid, length(map(:)), 'int32');
    fwrite(fid, map(:),  'single');
    fclose(fid);
    
    para0 = para0(:);
    save([des_path filename '.mat'], 'para0');
    
    
    fullname = [des_path filename '.input'];
    fullname = strrep(fullname, '\', '/');
    fprintf(ffid, [fullname, '\n']);
end

fclose(ffid);
