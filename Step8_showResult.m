%%% Showing the fitting results %%%

clear;
show_iter = 3;

load('Model_Expression.mat');
load('Model_Shape.mat');
load('params_attr.mat');

root_path = '..\Data\';
src_path = [root_path 'inputs\iter' num2str(show_iter) '\'];
img_path = [root_path 'samples\crop\'];

std_size = 100;
width = std_size;
height = std_size;
nChannels1 = 3;
nChannels2 = 3;

pca_co = 40;
pca_exp_co = 13;
para_nums = 234;

mu = mu_shape + mu_exp;
tex = tex / 255;

count = 0;

filelist = dir([src_path '*.input']);

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
    
    
    %% 1. Get the result parameters
    % load the regressed para_delta
    feat = [feat(1:5,:); zeros(1, size(feat,2)); feat(6:end,:)];

    std_mat = repmat(params_std, 1, size(feat,2));
    feat = feat .* std_mat;  
    shape_ind = 7+1:7+199; shape_void = shape_ind(pca_co+1:end);
    exp_ind = 7+199+1:7+199+29; exp_void = exp_ind(pca_exp_co+1:end);
    feat([shape_void, exp_void],:) = 0;
    parar = para0 + feat;
        
    %% 2. Show the fitting result
    para = parar;

    Pose_Para = para(1:7);
    Shape_Para = para(7+1:7+199);
    Exp_Para = para(7+199+1:end);
    
    [phi, gamma, theta, t3d, f] = ParaMap_Pose(Pose_Para);
    
    vertex = mu + w * Shape_Para + w_exp * Exp_Para;
    vertex = reshape(vertex, 3, length(vertex)/3);
    ProjectVertex= f * RotationMatrix(phi,gamma,theta) * vertex + repmat(t3d, 1, size(vertex,2));
    
    ProjectVertex(2,:) = height + 1 - ProjectVertex(2,:);
    result_img = Mex_ZBuffer(ProjectVertex, tri, tex, double(roi_img));
    
    
    
    subplot(1,2,1);
    imshow(roi_img);
    subplot(1,2,2);
    imshow((result_img*0.5 + roi_img*0.5));
    
end




