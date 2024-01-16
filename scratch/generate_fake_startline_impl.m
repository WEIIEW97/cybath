function mask = generate_fake_startline_impl(cx, cy, h, w, angle, bg_h, bg_w, scale)
%GENERATE_FAKE_STARTLINE Summary of this function goes here
%   Detailed explanation goes here
[x, y] = meshgrid(1:scale:bg_w*scale, 1:scale:bg_h*scale);

x = x - cx;
y = y - cy;

theta = deg2rad(angle); 
xRot = x*cos(theta) - y*sin(theta);
yRot = x*sin(theta) + y*cos(theta);

mask = abs(xRot) <= w/2 & abs(yRot) <= h/2;
end

