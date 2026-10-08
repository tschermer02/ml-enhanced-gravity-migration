%
%****************************************************************
%      PROGRAM Gravity & Gravity gradient MIGRATION-3D
% unit.... m s 6.67*e-11 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function gg_migration()
clear all;
close all;
clc;
%
disp('  3D Gravity & Gravity gradient 3D Migration V2.0  ');
disp('                CEMI         ');
disp('             2014-02-01           ');
c1=clock;
c2=[num2str(fix(c1(4))),':',num2str(fix(c1(5))),'''',num2str(fix(c1(6)))];
disp(['--- start :  ---  ',c2]);
%
input_gg_mig;     % Input data;
%
% Set parameters ---------------------
% filename = 'test1.dat';    %  input filename of gravity observed data
% migid=[1 2 3 4 5 6 7 8];
% modelid=0;
% % set the migration domain -- it also can read from file for x and y range
% xstep=50;
% ystep=50;
% mz = 30;      % mesh number of z direction
% zmin = 0;
% zmax = 1500; 
% delta_x = 50;
% delta_y = 50;
% restrictid = 1;
% ro_min = 0;
% ro_max = 1;
% end of set parameyer --------------------
%	check the input parameter
% find out if it need to restrict the result range
if(ro_min > ro_max )
    msgbox('Parameter set wrong! reset <ro_min> <ro_max>');
    return;
end
% 
%gamma = 6.67E-11*1E8;    % universal gravitational constant
%------------------
gg = load(filename);
% file format:
% xr, yr, zr, clearrance, gz, gxx, gxy, gxz, gyy, gyz, gzz, gdelta(=(gxx-gyy)/2)
% Topography = zr - clearrance
%------------------
[nrow,ncolumn] = size(gg);
nmig = ncolumn - 4;	   % gravity components number
if nmig < 1 	% check if the data colum correct
	msgbox('Please check if the data colum are correct.');
	return
end
if ~isempty(migjoint)
    joint=1;
    nj=length(migjoint);
    for j=1:nj
        indxj(j)=find(migjoint(j)==migid);
        if indxj(j)==0
            msgbox('Please check if the migjoint value.');
            return
        end
    end
else
    joint=0;
end
% 	the output file name is "rho" + "input filename"
outname=['den_',filename];
%
xr=gg(:,1);
yr=gg(:,2);
zr=-gg(:,3);	% change the recorder z-level down+
clearance=gg(:,4);
topo = zr + clearance;	% zr is "negtive" and clearance always positive
zstep=(zmax-zmin)/mz;
z=zeros(mz+1,1);
for i=1:mz+1
	z(i)=zmin+zstep*(i-1);
end
z_center=zeros(mz,1);
for i=1:mz
	z_center(i)=(z(i)+z(i+1))/2;
end
%	find out the migration domain
if modelid == 0
	xmin = xstep*fix(min(xr)/xstep)+xstep;
	xmax = xstep*ceil(max(xr)/xstep)-xstep;
	ymin = ystep*fix(min(yr)/ystep)+ystep;
	ymax = ystep*ceil(max(yr)/ystep)-ystep;
	x=xmin:xstep:xmax;
	mx=length(x)-1;
	y=ymin:ystep:ymax;
	my=length(y)-1;
	Nm=mx*my*mz;
    x_center=zeros(mx,1);
	for i=1:mx
		x_center(i)=(x(i)+x(i+1))/2;   %*xstep+(i-1)*xstep;
	end
    y_center=zeros(my,1);
	for i=1:my
		y_center(i)=(y(i)+y(i+1))/2;   %y(1)+0.5*ystep+(i-1)*ystep;
	end
%	form the migration domain coordinate
	[xx,yy,zz]=meshgrid(x_center,y_center,z_center);
    [xi,yi]=meshgrid(x_center,y_center);
elseif modelid == 1
	md=load(domainfile);
	xi=md(:,1);
	yi=md(:,2);
    xx=repmat(xi,[1,mz]);
    yy=repmat(yi,[1,mz]);
	Nxy=length(xi);
    Nm=Nxy*mz;
    zz=(repmat(z_center,[1,Nxy]))';
else
    msgbox('Please check the input parameter ''modelid''. it shoud be 0 or 1.');
    return;
end
xm=reshape(xx,Nm,1);
ym=reshape(yy,Nm,1);
zm=reshape(zz,Nm,1);
% 	interpolation
chr=version;
if chr(4)=='.'
	vn=str2double(chr(1:3));
else
	vn=str2double(chr(1:4));
end
if vn > 7.9
%  For MATLAB 2010 and above
    Fm=TriScatteredInterp(xr,yr,topo);
    mtopo=Fm(xi,yi);			% used for shift the migration domai to the topography
    Fw=TriScatteredInterp(xr,yr,clearance);
    mclearance=Fw(xi,yi);		% add to the zm will used for weighting
else
%  for MATLAB 2006 and  old version
    mtopo=griddata(xr,yr,topo,xi,yi);
    mclearance=griddata(xr,yr,clearance,xi,yi);
end
zwm = repmat(mclearance,[1,1,mz]);  
zwm = reshape(zwm,Nm,1);
zwm = zwm + zm;
dzm = repmat(mtopo,[1,1,mz]); 
dzm = reshape(dzm,Nm,1);
zm=zm+dzm;	% shift to the topography
% set a matrix to store the result
dms=zeros(Nm,nmig+3);
dms(:,1)=xm;
dms(:,2)=ym;
dms(:,3)=zm;
c1=clock;
c2=[num2str(fix(c1(4))),':',num2str(fix(c1(5))),'''',num2str(fix(c1(6)))];
disp(['--- ready for migration --- ',c2]);
for k=1:nmig
    vr=gg(:,k+4);
    [dm] = migrationfunc(migid(k),xr,yr,zr,vr,zwm,xm,ym,zm);
	if restrictid == 1
		zoom=(ro_max-ro_min)./(max(dm)-min(dm));
		dm=(dm-min(dm)).*zoom+ro_min;
	end
    dms(:,3+k) = dm;
end
% save data
save(outname,'dms','-ascii');
if joint==1
    dmj=zeros(Nm,4);
    dj(:,1:nj)=dms(:,indxj+3);
    dmj(:,1:3)=dms(:,1:3);
    dmj(:,4)=sum(dj,2)./nj;
    outname2=['denj_',filename];
    save(outname2,'dmj','-ascii');
end
return;
%
% ---------------------------------------------------------
%
function [dm] = migrationfunc(migid,xr,yr,zr,vr,zwm,xm,ym,zm)
Nd=length(xr);
Nm=length(xm);
dm=zeros(Nm,1);
%delta_x = 20;
%delta_y = 20;
gamma = 6.67E-11*1E8;    % universal gravitational constant
switch migid
    case 1;
        c0='== migration Gz --->';
	case 2;
        c0='== migration Gxx --->';
    case 5;
        c0='== migration Gyy --->';
    case 7;
        c0='== migration Gzz --->';
    case 3;
        c0='== migration Gxy --->';
    case 4;
        c0='== migration Gxz --->';
    case 6;
        c0='== migration Gyz --->';
    case 8;
        c0='== migration Gdelta --->';
    otherwise
        disp('don''t know what to do.');
        return
end
c1=clock;
c2=[num2str(fix(c1(4))),':',num2str(fix(c1(5))),'''',num2str(fix(c1(6)))];
disp([c0,' ',c2]);     
for ir=1:Nd
    xxd=xr(ir)-xm(:);
    yyd=yr(ir)-ym(:);
    zzd=zm(:)-zr(ir);  %  down(+)
    xxd2=xxd.^2;
    yyd2=yyd.^2;
    zzd2=zzd.^2;
    r2 = xxd2+yyd2+zzd2;  
    r=sqrt(r2);
    r3=r.*r2;
    r5=r3.*r2;
    switch migid
        case 1;
            gzm = vr(ir).*zzd ./r3;                     % Gz 
			dm=dm+gzm;
        case 2;
            gxx = vr(ir).*(2*xxd2 - yyd2 - zzd2)./r5; 	% Gxx
			dm=dm+gxx;
        case 5;
            gyy = vr(ir).*(2*yyd2 - xxd2 - zzd2)./r5;	% Gyy
			dm=dm+gyy;
        case 7;
            gzz = vr(ir).*(2*zzd2 - xxd2 - yyd2)./r5;	% Gzz
			dm=dm+gzz;
        case 3;
            gxy = vr(ir).*xxd.*yyd./r5;                 % Gxy
			dm=dm+gxy;
        case 4;
            gxz = vr(ir).*xxd.*zzd./r5;     	   		% Gxz
			dm=dm+gxz;
        case 6;
            gyz = vr(ir).*yyd.*zzd./r5;                 % Gyz
			dm=dm+gyz;
        case 8;
            gdelta = vr(ir).*(xxd2-yyd2)./r5;           % Gdelta
			dm=dm+gdelta;
        otherwise
            disp('don''t know what to do.');
            return
    end
    if(mod(ir,fix(Nd/20))==0) 
        c0=['--- ' num2str(ceil(100*ir/Nd)) '% finished ---'];
        c1=clock;
c2=[num2str(fix(c1(4))),':',num2str(fix(c1(5))),'''',num2str(fix(c1(6)))];
        disp([c0,' ',c2]);     
    end
end
switch migid
    case 1;
		cz=gamma*sqrt(pi);
        dm=abs(zwm).*dm./cz;
	case 2;
		cxx=gamma*3*sqrt(pi)/4;
        dm=(zwm.^2).*dm./cxx;
    case 5;
		cyy=gamma*3*sqrt(pi)/4;
        dm=(zwm.^2).*dm./cyy;
    case 7;
		czz=gamma*sqrt(3*pi)/2;
        dm=(zwm.^2).*dm./czz;
    case 3;
		cxy=gamma*sqrt(pi)/4;
        dm=(zwm.^2).*dm./cxy;
    case 4;
        cxz=gamma*sqrt(3*pi)/6;
        dm=(zwm.^2).*dm/cxz;
    case 6;
        cyz=gamma*sqrt(3*pi)/6;
        dm=(zwm.^2).*dm./cyz;
    case 8;
        cdelta=gamma*sqrt(pi)/4;
        dm=(zwm.^2).*dm./cdelta;
    otherwise
        disp('don''t know what to do.');
        return
end
return;
% **************************END************************************