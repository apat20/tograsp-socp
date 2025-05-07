% Script to effectively visualize the results and output of our neural
% network based approach. 

clear all;
close all;
clc;

addpath('matlab_helper_funcs/')

% Adding data path:
addpath('logs/pickup_trial_1/')

% Reading the point cloud files for the CheezIt box:
% Point transformed to the object reference frame:
ptCloud1 = pcread('initial_cloud_object_frame.ply');

% Point cloud on one of the faces of the bounding box:
ptCloud2 = pcread('projected_object_frame.ply');

% Extract xyz coordinates of points from the transformed point cloud. 
XYZpoints_object_frame = reshape(ptCloud1.Location, [],3);

% Extract xyz coordinates of points from the projected point cloud. 
XYZpoints_projected = reshape(ptCloud2.Location, [],3);

% Object Reference frame at the initial location when we load it:
R_Object_initial = eye(3); p_Object_initial = zeros(3,1);

% Reading the bounding box coordinates:
Vertices = importdata("transformed_vertices_object_frame.csv");

% Faces of the bounding box constructed using its vertices:
Faces = [2 1 4 7; 1 3 6 4; 3 8 5 6; 8 2 7 5];

% Reading the center of the bounding box:
Center = [(Vertices(4,1) + Vertices(7,1))/2;(Vertices(6,2) + Vertices(7,2))/2; (Vertices(6,3) + Vertices(3,3))/2];

% Getting the dimensions of the box in terms of the X, Y and Z directions:
x_L = abs(Vertices(1,1) - Vertices(2,1));
y_W = abs(Vertices(3,2) - Vertices(1,2));
z_H = abs(Vertices(4,3) - Vertices(1,3));

%% Loading the saved results:

datapoints = csvread("test_datapoints.csv");
labels = importdata("test_predicted.csv");

% Extracting the screw parameters from the loaded datapoints: 
screw_axis = importdata("unit_vector.csv");
% Reading and loading up the point corresponding to the screw axis:
point = importdata("point.csv");

X_grid_points = importdata('X_grid_points_computed.csv');
Y_grid_points = importdata('Y_grid_points_computed.csv');
Z_grid_points = importdata('Z_grid_points_computed.csv');

X_grid_points_occupied = importdata('X_grid_points_occupied_computed.csv');
Y_grid_points_occupied = importdata('Y_grid_points_occupied_computed.csv');
Z_grid_points_occupied = importdata('Z_grid_points_occupied_computed.csv');
 
metric_grid = importdata("metric_grid_computed.csv");
grid_metric_values_occupied = importdata("metric_grid_occupied_computed.csv");

%% Visualizing the results:

figure()
grid minor
hold on;
xlabel('X')
hold on;
ylabel('Y')
hold on;
zlabel('Z')
hold on;
plot_frame(R_Object_initial, p_Object_initial, 0.02);
hold on;
pcshow(ptCloud1, 'MarkerSize', 12, 'BackgroundColor',[1,1,1]);
hold on;
% patch('Faces',Faces,'Vertices',Vertices,'FaceColor','none','EdgeColor','r','LineWidth',1);
% hold on;
q1 = quiver3(point(1), point(2), point(3), screw_axis(1), screw_axis(2), screw_axis(3), 0.1, 'Color','r', 'AutoScale','off', 'linewidth',2);
hold on;
plot3(point(1), point(2), point(3),'-o','Color','r','MarkerSize',10);
hold on;
axis off;

figure()
grid minor
hold on;
xlabel('X')
hold on;
ylabel('Y')
hold on;
zlabel('Z')
hold on;
plot_frame(R_Object_initial, p_Object_initial, 0.02);
hold on;
pcshow(ptCloud2, 'MarkerSize', 12, 'BackgroundColor',[1,1,1]);
hold on;
patch('Faces',Faces,'Vertices',Vertices,'FaceColor','none','EdgeColor','r','LineWidth',1);
hold on;
axis off;

% Visualizing the entire grid: 
Y_grid_points_transpose = Y_grid_points.';
X_grid_points_transpose = X_grid_points.';
Z_grid_points_transpose = Z_grid_points.';

figure()
grid minor
hold on;
xlabel('X')
hold on;
ylabel('Y')
hold on;
zlabel('Z')
hold on;
plot_frame(R_Object_initial, p_Object_initial, 0.02);
hold on;
plot_frame(R_Object_initial, Center, 0.02);
hold on;
pcshow(ptCloud2, 'MarkerSize', 12, 'BackgroundColor',[1,1,1]);
hold on;
patch('Faces',Faces,'Vertices',Vertices,'FaceColor','none','EdgeColor','r','LineWidth',1);
hold on;
fill3(X_grid_points_transpose, Y_grid_points_transpose, Z_grid_points_transpose, metric_grid);
hold on;
% plot3(XYZpoints_projected(1,1), XYZpoints_projected(1,2), XYZpoints_projected(1,3),'o','Color','r','MarkerSize', 25);
% hold on;
cb1 = colorbar;
axis off;


% % Visualizing the occupied grid: 
Y_grid_points_occupied_transpose = Y_grid_points_occupied.';
X_grid_points_occupied_transpose = X_grid_points_occupied.';
Z_grid_points_occupied_transpose = Z_grid_points_occupied.';

figure()
grid minor
hold on;
xlabel('X')
hold on;
ylabel('Y')
hold on;
zlabel('Z')
hold on;
plot_frame(R_Object_initial, p_Object_initial, 0.02);
hold on;
plot_frame(R_Object_initial, Center, 0.02);
hold on;
pcshow(ptCloud2, 'MarkerSize', 12, 'BackgroundColor',[1,1,1]);
hold on;
patch('Faces',Faces,'Vertices',Vertices,'FaceColor','none','EdgeColor','r','LineWidth',1);
hold on;
fill3(X_grid_points_occupied_transpose, Y_grid_points_occupied_transpose, Z_grid_points_occupied_transpose, grid_metric_values_occupied);
hold on;
% plot3(XYZpoints_projected(1,1), XYZpoints_projected(1,2), XYZpoints_projected(1,3),'o','Color','r','MarkerSize', 25);
% hold on;
cb2 = colorbar;
axis off;

% Visualizing the actual point cloud with assigned values: 
figure()
hold on;
xlabel('X')
hold on;
ylabel('Y')
hold on;
zlabel('Z')
hold on;
pcshow(XYZpoints_object_frame, grid_metric_values_occupied ,'MarkerSize', 12, 'BackgroundColor',[1,1,1]);
hold on;
% patch('Faces',Faces,'Vertices',Vertices,'FaceColor','none','EdgeColor','r','LineWidth',1);
% hold on;
% q1 = quiver3(P(1), P(2), P(3), screw_axis(1), screw_axis(2), screw_axis(3), 0.08, 'Color',[0.4940 0.1840 0.5560] , 'AutoScale','off');
% hold on;
% plot3(P(1), P(2), P(3),'-o','Color','b','MarkerSize',10);

% %% Assigning new grid metric values and recomputing:
% for i = 1:size(grid_metric_values_occupied, 1)
%     metric_value = grid_metric_values_occupied(i, :);
%     if metric_value == 0
%         new_metric_value = grid_metric_values_occupied(i+1, :);
%         grid_metric_values_occupied(i, :) = new_metric_value;
%     end
% end
% 
% % Visualizing the actual point cloud with assigned values: 
% figure()
% hold on;
% xlabel('X')
% hold on;
% ylabel('Y')
% hold on;
% zlabel('Z')
% hold on;
% pcshow(XYZpoints_object_frame, grid_metric_values_occupied ,'MarkerSize', 12, 'BackgroundColor',[1,1,1]);
% hold on;
% axis off;
% % patch('Faces',Faces,'Vertices',Vertices,'FaceColor','none','EdgeColor','r','LineWidth',1);
% % hold on;
% % q1 = quiver3(P(1), P(2), P(3), screw_axis(1), screw_axis(2), screw_axis(3), 0.08, 'Color',[0.4940 0.1840 0.5560] , 'AutoScale','off');
% % hold on;
% % plot3(P(1), P(2), P(3),'-o','Color','b','MarkerSize',10);



