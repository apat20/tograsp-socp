%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is used to plot a reference frame corresponding to a
% configuration (rotation matrix and a position vector).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By: Aditya Patankar

function plot_frame(R, p, scale)
%     scale = 0.02;
%     axis equal;
%     q1 = quiver3(p(1),p(2),p(3), R(1,1), R(2,1), R(3,1),scale, 'Color','red', 'AutoScale','off');
    q1 = quiver3(p(1),p(2),p(3), R(1,1), R(2,1), R(3,1),scale, 'Color','red', 'AutoScale','off', 'linewidth',5);
    hold on;
    q2 = quiver3(p(1),p(2),p(3), R(1,2), R(2,2), R(3,2),scale,'Color','green', 'AutoScale','off','linewidth',5);
    hold on;
    q3 = quiver3(p(1),p(2),p(3), R(1,3), R(2,3), R(3,3),scale,'Color','blue','AutoScale','off', 'linewidth',5);
    hold off;

end
