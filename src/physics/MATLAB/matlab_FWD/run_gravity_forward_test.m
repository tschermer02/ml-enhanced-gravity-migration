function run_gravity_forward_test(input_dir, output_dir)
% Run an independent MATLAB cell-centered Gz forward calculation on every
% exported test model. Density units: g/cm^3; Gz output: mGal.
% The exported density is flattened in x-fastest order, matching the Python
% C-order layout used by export_gravity_test_set.py and by the Python validation
% script compare_gravity_forward.py.
% Usage:
%   run_gravity_forward_test('analysis_outputs/gravity_forward_validation/inputs', ...
%                           'analysis_outputs/gravity_forward_validation/matlab_outputs')

if nargin < 1
    input_dir = fullfile('analysis_outputs', 'gravity_forward_validation', 'inputs');
end
if nargin < 2
    output_dir = fullfile('analysis_outputs', 'gravity_forward_validation', 'matlab_outputs');
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

index = readtable(fullfile(input_dir, 'index.csv'), 'TextType', 'string');
G = 6.67e-11;
scale = 1e8;
h = 10.0;
[ox, oy] = meshgrid(-85:10:715, -85:10:715);
rx = ox(:);
ry = oy(:);
n_receivers = numel(rx);

for sample_idx = 1:height(index)
    sid = char(index.sample_id(sample_idx));
    filename = fullfile(input_dir, char(index.h5_name(sample_idx)));

    rho_vec = h5read(filename, '/density_xfast_gcm3');
    % Exported density is C-order x-fastest: (z, y, x) flattened as [z0y0x0, z0y0x1, ...].
    % Reshape to [64,64,24] so index order is [x, y, z].
    rho = reshape(rho_vec, [64, 64, 24]);

    [ix, iy, iz] = ind2sub(size(rho), find(rho ~= 0));
    rho_active = rho(rho ~= 0);

    x = (double(ix) - 0.5) * h;
    y = (double(iy) - 0.5) * h;
    z = (double(iz) - 0.5) * h;
    weights = G * scale * h^3 * rho_active;

    gz = zeros(n_receivers, 1);
    batch_size = 128;
    for s = 1:batch_size:n_receivers
        e = min(s + batch_size - 1, n_receivers);
        dx = x.' - rx(s:e);
        dy = y.' - ry(s:e);
        dz = z.';
        r2 = dx.^2 + dy.^2 + dz.^2;
        gz(s:e) = sum(weights.' .* dz ./ (r2 .* sqrt(r2)), 2);
    end

    % MATLAB stores the field in the same flattened ordering the Python
    % comparison expects: (81x81) map transposed to x-fastest vector order.
    gz_xy = reshape(gz, [81, 81]);
    matlab_gz_xfast_mgal = reshape(gz_xy.', [], 1);
    save(fullfile(output_dir, [sid '.mat']), 'matlab_gz_xfast_mgal', '-v7');
    fprintf('MATLAB %d/%d: %s\n', sample_idx, height(index), sid);
end
end
