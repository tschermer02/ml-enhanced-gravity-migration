function run_migration_comparison()
% Run gg_migration for every staged gravity map in case_ids.txt.
% The working directory contains gg_migration.m, input_gg_mig.m,
% domain.dat, and one <sample_id>.dat observation file per case.

fid = fopen('case_ids.txt', 'r');
if fid < 0
    error('Could not open case_ids.txt');
end
case_ids = textscan(fid, '%s');
fclose(fid);
case_ids = case_ids{1};

for sample_idx = 1:numel(case_ids)
    fid = fopen('current_case.txt', 'w');
    if fid < 0
        error('Could not write current_case.txt');
    end
    fprintf(fid, '%s', case_ids{sample_idx});
    fclose(fid);

    gg_migration();
    fprintf('MATLAB migration %d/%d: %s\n', ...
        sample_idx, numel(case_ids), case_ids{sample_idx});
end
end
