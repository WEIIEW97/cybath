function  generate_fake_startline()
%GENERATE_FAKE_STARTLINE Summary of this function goes here
%   Detailed explanation goes here
h = 10;
w = 600;
angle_range = [-15, 15];

bg_h = 480;
bg_w = 640;

scale = 1;

savepath = "/home/william/Codes/find-landmark/data/start_line/fake";

for i = 1:10
    cx = randi([bg_w/2-30, bg_w/2+30]);
    cy = randi([bg_h-60, bg_h-10]);
    angle = randi(angle_range);
    mask = generate_fake_startline_impl(cx, cy, h, w, angle, bg_h, bg_w, scale);
    ext_name = 'start_%04d.jpg';
    full_name = sprintf(savepath+'/'+ext_name, i);
    imwrite(mask, full_name);
end
end

