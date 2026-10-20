@echo off
rem The results zip a site sends back (see export_results.py next to this file, which does the work).
rem
rem   usage: export_results.cmd [--allow-small-stage-totals] [study ...]
rem
rem   --allow-small-stage-totals  export although a stage excluded 1 to min_cell - 1
rem                               records in total; only after the site decided that
rem                               count may leave (recorded in the bundle)
rem
rem   MIVAL_HOME  bundle root (default: the directory above this repository)
rem   MIVAL_RUNS  where run artifacts go (default: %MIVAL_HOME%\runs)
rem
rem This file only finds the mival-torch interpreter; Linux uses the .sh file
rem the same way, so both run the same Python steps.
setlocal
rem Child logs and Korean messages are UTF-8; a redirected cp949 stream would garble them.
if not defined PYTHONIOENCODING set "PYTHONIOENCODING=utf-8"
rem This script lives in scripts\ in the shared copy and in scripts\site\ in mi-va.
set "REPO=%~dp0.."
if not exist "%REPO%\src\mival" set "REPO=%~dp0..\.."
if not defined MIVAL_HOME for %%I in ("%REPO%\..") do set "MIVAL_HOME=%%~fI"
if not exist "%MIVAL_HOME%\envs\mival-torch\python.exe" (
  echo no %MIVAL_HOME%\envs\mival-torch\python.exe ^(see README: env install^)
  exit /b 1
)
"%MIVAL_HOME%\envs\mival-torch\python.exe" "%~dp0export_results.py" %*
exit /b %ERRORLEVEL%
