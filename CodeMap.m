function [ code_map ] = CodeMap(mu, w, w_exp, tri, vertex_code, img, Pose_Para, Shape_Para, Exp_Para)
[phi, gamma, theta, t3d, f] = ParaMap_Pose(Pose_Para);
R = RotationMatrix(phi, gamma, theta);
P = [1 0 0; 0 1 0];

alpha = Shape_Para;
alpha_exp = Exp_Para;
express = w_exp * alpha_exp; express = reshape(express, 3, length(express)/3);
shape = mu + w * alpha; shape = reshape(shape, 3, length(shape)/3);
vertex = shape + express;

ProjectVertex = f * R * vertex + repmat(t3d, 1, size(vertex, 2));

%DrawSolidHead(ProjectVertex, tri);
[height, width, nChannels] = size(img);
code_map = zeros(height, width, nChannels);
[code_map] = Mex_ZBuffer(double(ProjectVertex), tri, double(vertex_code), code_map);
code_map = InvertImage(code_map);
end

