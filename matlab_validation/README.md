Run at repository root. Copy prepare_validation.py and compare_validation.py to root (or execute with a path), and run_matlab_validation.m to root.
python prepare_validation.py --limit 100
matlab -batch "run_matlab_validation('outputs/ml/validation/matlab_D01')"
python compare_validation.py --limit 100
MATLAB uses FFT-based full convolution rather than original nested receiver-model loops, but implements the identical point-pulse kernels from getPredGR.m cases 4,7 and gg_migration2014.m migid 1,7 on the flat fixed grid. It is a numerical cross-language verification of equations, not a run of unmodified original scripts. No noise in forward and migration tests; all density volumes from D01 test. CPU-intensive; run on allocated compute node. May take significant time, especially the 100 volumes; consider running 1 then 100.
