function  generate_fake_startline()
%GENERATE_FAKE_STARTLINE Summary of this function goes here
%   Detailed explanation goes here
h = 10;
w = 600;
max_angle = 30;

bg_h = 480;
bg_w = 640;

scale = 1;

savepath = "/home/william/Codes/find-landmark/data/start_line/fake";

for i = 1:10
    cx = randi(bg_w);
    cy = randi(bg_h);
    angle = randi(max_angle);
    mask = generate_fake_startline_impl(cx, cy, h, w, angle, bg_h, bg_w, scale);
    ext_name = 'start_%04d.jpg';
    full_name = sprintf(savepath+'/'+ext_name, i);
    imwrite(mask, full_name);
end
end

