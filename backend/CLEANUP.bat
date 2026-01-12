@echo off
echo Cleaning up workspace...

:: Create archive directory
if not exist "archive" mkdir archive

:: Move debug scripts
move debug_*.py archive\
move inspect_apis.py archive\
move run_demo_robust.py archive\
move run_trial_v2.py archive\

:: Move log files
move *log.txt archive\

:: Move old output artifacts
move final_output.jpg archive\
move output_vton.jpg archive\
move result.jpg archive\
move test_output_1522.jpg archive\

:: Move unused model file
move age_net_test.caffemodel archive\

echo Cleanup complete!
pause
