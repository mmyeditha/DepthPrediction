close all
weights = [];
ptCloud = readmatrix('cloud_2226.csv');
% lidarCloud = readmatrix('lidar_1325.csv');

R = [0 1 0 0; -1 0 0 0; 0 0 1 0; 0 0 0 1];
Tyz = [-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
Txz = [1 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 1];

ptCloud = ((ptCloud*R)*Tyz)*Txz;

% maxDepth = 3;
% ptCloud = ptCloud(ptCloud(:,4)<= maxDepth,:);


figure;
pcshow(ptCloud(:,1:3));
xlabel('x'); ylabel('y'); zlabel('z');
axis equal;

pointsScaled = ptCloud.*ptCloud(:,4);

% Doing it with non-lidar
% figure;
% scatter3(pointsScaled(:,1), pointsScaled(:,2), pointsScaled(:,3),1);
xlabel('x'); ylabel('y'); zlabel('z');
axis equal
figure;
pcshow(ptCloud(:,1:3).*ptCloud(:,4));
xlabel('x'); ylabel('y'); zlabel('z');
axis equal;

% Now with lidar

% maxDepth = 3;
% lidarCloud = lidarCloud(lidarCloud(:,4)<= maxDepth,:);
% pointsScaled = lidarCloud.*lidarCloud(:,4);
% 
% % figure;
% % scatter3(pointsScaled(:,1), pointsScaled(:,2), pointsScaled(:,3),1);
% xlabel('x'); ylabel('y'); zlabel('z');
% axis equal
% figure;
% pcshow(lidarCloud(:,1:3).*lidarCloud(:,4));
% xlabel('x'); ylabel('y'); zlabel('z');
% axis equal;
