filename = 'test2.dat';    %  input filename of gravity observed data
migid=[1 2 3 4 5 6 7 8];   %  indicator of component.
%   1 -- Gz -- the vertical component of the gravity field (mGal),
%   2 -- Gxx -- a component of gravity gradiometry data (Eotvos),
%   3 -- Gxy,
%   4 -- Gxz,
%   5 -- Gyy,
%   6 -- Gyz,
%   7 -- Gzz
%   8 -- Gdelta =(Gxx-Gyy)/2
migjoint=[2 5 7];
modelid=0;                 %  indicator for migration domain information.
%domainfile='test0_md.dat'; %  x and y coordinates of migration domain.
% set the migration domain -- it also can read from file for x and y range.
xstep=100;                 %  migration domain cell size in x direction.
ystep=100;                 %  migration domain cell size in y direction.
mz = 30;                   %  mesh number of z direction.
zmin = 0;                  %  The minimum value of z depth of the migration domain (m).
zmax = 1500;               %  The maximum value of z depth of the migration domain (m).
delta_x = 50;              %  observed data interval in x direction (m).
delta_y = 50;              %  observed data interval in y direction (m).
restrictid = 1;            %  indicator that if need to restrict the output density.
ro_min = 0;                %  minimum value to restrict the output density.
ro_max = 1;                %  maximum value to restrict the output density.

