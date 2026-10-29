function run_matlab_validation(root)
% Independent MATLAB implementation of getPredGR (cases 4,7) and
% gg_migration2014 (migid 1,7) on the fixed 10 m, 24x64x64 grid.
% Uses FFT convolution for speed; same discrete point-source kernels.
if nargin < 1, root = fullfile('outputs','ml','validation','matlab_D01'); end
in_dir=fullfile(root,'inputs');out_dir=fullfile(root,'matlab');
if ~exist(out_dir,'dir'),mkdir(out_dir);end
files=dir(fullfile(in_dir,'sample_*.mat'));
if isempty(files),error('No MATLAB inputs in %s',in_dir);end
G=6.67e-11; h=10; gamma=G*1e8;
off=(-72:71)*h; [dx,dy]=meshgrid(off,off);
for n=1:numel(files)
    sample=load(fullfile(in_dir,files(n).name));rho=double(sample.density);
    if ~isequal(size(rho),[24 64 64]),error('Density shape incorrect');end
    gz_forward=zeros(81,81);gzz_forward=zeros(81,81);
    for iz=1:24
        layer=squeeze(rho(iz,:,:));
        if ~any(layer(:)),continue;end
        z=(iz-0.5)*h;r2=dx.^2+dy.^2+z^2;
        k_gz=gamma*h^3*z./(r2.*sqrt(r2));
        k_gzz=G*1e12*h^3*(3*z^2-r2)./(r2.^2.*sqrt(r2));
        % Python scipy fftconvolve full crop [63:144] (0-based).
        a=fftconv2_full(layer,k_gz);gz_forward=gz_forward+a(64:144,64:144);
        a=fftconv2_full(layer,k_gzz);gzz_forward=gzz_forward+a(64:144,64:144);
    end
    gz_migration=zeros(24,64,64);gzz_migration=zeros(24,64,64);
    for iz=1:24
        z=(iz-0.5)*h;r2=dx.^2+dy.^2+z^2;
        k_gz=z./(r2.*sqrt(r2));
        k_gzz=(3*z^2-r2)./(r2.^2.*sqrt(r2));
        a=fftconv2_full(gz_forward,rot90(k_gz,2));
        gz_migration(iz,:,:)=a(81:144,81:144)*z/(gamma*sqrt(pi));
        a=fftconv2_full(gzz_forward,rot90(k_gzz,2));
        gzz_migration(iz,:,:)=a(81:144,81:144)*z^2/(gamma*sqrt(3*pi)/2);
    end
    save(fullfile(out_dir,files(n).name),'gz_forward','gzz_forward',...
        'gz_migration','gzz_migration','-v7');
    fprintf('MATLAB %d/%d: %s\n',n,numel(files),files(n).name);
end
end

function out=fftconv2_full(a,b)
sz=size(a)+size(b)-1;
out=real(ifft2(fft2(a,sz(1),sz(2)).*fft2(b,sz(1),sz(2))));
end
