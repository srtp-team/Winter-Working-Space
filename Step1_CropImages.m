%%% Cropping roi image from bounding box %%%

clear;
root_path = '..\Data\';

image_path = [root_path 'samples\images\'];
des_path = [root_path 'samples\crop\'];

%load filelist
load('vertex_code');
load('Model_Expression.mat');
load('Model_Shape.mat');
load('BaseSample.mat');

std_size = 100;

vertex_mean = reshape(mu_shape, 3, length(mu_shape)/3);
mu = mu_shape + mu_exp;

base_ind = keypoints;
base_ind1 = [(3 * base_ind - 2); (3 * base_ind -1); (3 * base_ind)]; 
base_ind1 = base_ind1(:);
mu_base = mu(base_ind1);
w_base = w(base_ind1,:);
w_exp_base = w_exp(base_ind1,:);

filelist = dir([image_path '*.jpg']);

for fi = 1:length(filelist)
    fi
    filename = filelist(fi).name(1:end-4);
    img = imread([image_path filename '.jpg']);
    load([image_path filename '.mat']);
    [height, width, nChannels] = size(img);
    
    
    % Reconstruct 3D point of base point
    project_base_point = pt2d_3d;
    
    bbox = [min(project_base_point(1,:)), min(project_base_point(2,:)), max(project_base_point(1,:)), max(project_base_point(2,:))];
    
    center = [(bbox(1)+bbox(3))/2, (bbox(2)+bbox(4))/2];
    radius = max(bbox(3)-bbox(1), bbox(4)-bbox(2)) / 2;
    bbox = [center(1) - radius, center(2) - radius, center(1) + radius, center(2) + radius];
    

    bbox = double(bbox);
    widthb = vertex_mean(1,keypoints(17)) - vertex_mean(1,keypoints(1));
    heightb = vertex_mean(2, keypoints(20)) - vertex_mean(2, keypoints(9));

    mean_x = vertex_mean(1, keypoints(31));
    mean_y = vertex_mean(2, keypoints(31));

    f0 = ((bbox(3) - bbox(1)) / widthb + (bbox(4) - bbox(2)) / heightb) / 2;
    t3d0(1) = (bbox(3) + bbox(1))/2 - f0*mean_x;
    temp = height + 1 - (bbox(2) + bbox(4))/2;
    t3d0(2) = temp - f0 * mean_y;
    t3d0(3) = 0;

    Pose_Para0 = [0, 0, 0, t3d0(1), t3d0(2), t3d0(3), f0];
    Shape_Para0 = zeros(size(Shape_Para));
    Exp_Para0 = zeros(size(Exp_Para));

    llength = sqrt((bbox(3)-bbox(1)) * (bbox(3)-bbox(1)) + (bbox(4)-bbox(2)) * (bbox(4)-bbox(2)));
    llength = round(llength * 1.2);
    center_x = (bbox(3) + bbox(1))/2;
    center_y = (bbox(4) + bbox(2))/2;
    roi_box(1) = round(center_x - llength / 2);
    roi_box(2) = round(center_y - llength / 2);
    roi_box(3) = roi_box(1) + llength;
    roi_box(4) = roi_box(2) + llength;


    roi_box = round(roi_box);
    [roi_img] = ImageROI(img, roi_box);
    roi_img = imresize(roi_img, [std_size, std_size]);
    [heightr, widthr, nChannels] = size(roi_img);
%    imshow(roi_img/255);

    roi_scale = std_size / double(roi_box(3)-roi_box(1));
    Pose_Para0_roi = Pose_Para0;
    Pose_Para0_roi(7) = Pose_Para0_roi(7) * roi_scale;

    Pose_Para0_roi(4) = (Pose_Para0_roi(4) - roi_box(1) + 1) * roi_scale;
    temp = (height + 1 - Pose_Para0_roi(5) - roi_box(2)) * roi_scale;
    temp = heightr + 1 - temp;
    Pose_Para0_roi(5) = temp;


    para0 = [Pose_Para0_roi, Shape_Para0', Exp_Para0'];
    parag = para0;
    c3d_map = CodeMap(mu, w, w_exp, tri, vertex_code, roi_img, Pose_Para0_roi, Shape_Para0, Exp_Para0);

    roi_img = single(roi_img);
    c3d_map = single(c3d_map);
    
%     imshow(roi_img/255);
%     imshow(c3d_map);
    save([des_path 'AFLW_' filename '.mat'], 'para0', 'parag', 'roi_img', 'c3d_map', 'roi_box');
end
