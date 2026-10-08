%  
%  Show the vertical cross-section from x, y, z, density
%
% ----------input parameter---------------
%
% filename -- input file name
%
% ----------output----------------------
%
%   a figure
%
filename='test1';               % original data
filename1=[filename,'.dat'];
j=1;    
% 1~8
k=j+4;   
migid=[9];
switch migid(j)
    case 1;
        myylabel='Gz (mGal)';
	case 2;
        myylabel='Gxx (E)';
    case 5;
        myylabel='Gyy (E)';
    case 7;
        myylabel='Gzz (E)';
    case 3;
        myylabel='Gxy (E)';
    case 4;
        myylabel='Gxz (E)';
    case 6;
        myylabel='Gyz (E)';
    case 8;
        myylabel='G\Delta (E)';
    case 9;
        myylabel='Gjoint (E)';
    otherwise
        disp('don''t know what to do.');
        return
end
ggdata = load(filename1);
xm = ggdata(:,1);
ym = ggdata(:,2);
vm = ggdata(:,k);
yc=fix(length(unique(ym))/2);
ind=find(ym==ym(yc));
xc0=xm(ind);
vc0=vm(ind);
% to noise data file
% filename2=[filename,'n20.dat']; % noise data
% ggdata = load(filename2);
% xm = ggdata(:,1);
% ym = ggdata(:,2);
% vm = ggdata(:,5);
% yc=fix(length(unique(ym))/2);
% ind=find(ym==ym(yc));
% xc1=xm(ind);
% vc1=vm(ind);

filename3=['denj_',filename1];  % density distribution
desdata = load(filename3);
xm = desdata(:,1);
ym = desdata(:,2);
zm = desdata(:,3);
vm = desdata(:,k-1);
yc=fix(length(unique(ym))/2);
ind=find(ym==ym(yc));
xc=xm(ind);
zc=zm(ind);
vc=vm(ind);
nx=length(unique(xc));
nz=length(xc)/nx;
xc2=reshape(xc,nx,nz);
zc2=reshape(zc,nx,nz);
vc2=reshape(vc,nx,nz);

    figure(7)
    subplot(2,1,2)
    contourf(xc2,zc2,vc2,20,'LineStyle','none');
    hold on
    axis equal;
    axis ij;
    set(gca,'FontWeight','Bold','FontSize',12);    %
%    title(['migration model'],'FontWeight','Bold','FontSize',14);
    %set(gca,'YDir','Reverse');           % reverse the y coordinate
    xlabel('x(m)','FontWeight','Bold','FontSize',12);
    ylabel('z(m)','FontWeight','Bold','FontSize',12);
%    axis([min(xc) max(xc) min(zc) max(zc)]);
    axis([min(xc) max(xc) min(zc) 1200]);
plot([-250 250 250 -250 -250],[100 100 600 600 100],'w-');
%hold on
%plot([250 750 750 250 250],[200 200 700 700 200],'w-');
    %caxis([ro_min,ro_max]);   %set colorbar limits
    h=colorbar('vert','FontWeight','Bold','FontSize',12);
    set(get(h,'title'),'string','\rho(g/m^3)','FontWeight','Bold','FontSize',11); 
    hold off
    
    subplot(2,1,1)
%    plot(xc0,vc0,'r-',xc1,vc1,'b-','LineWidth',2);
    plot(xc0,vc0,'r-','LineWidth',2);
    set(gca,'FontWeight','Bold','FontSize',12);    %
    ylabel(myylabel,'FontWeight','Bold','FontSize',12);
    colorbar('vert','FontWeight','Bold','FontSize',12);
    grid on
    