@echo off
setlocal

if "%~1"=="" (
    :: Default: yesterday's date with --llm --verbose
    for /f "tokens=1-3 delims=-" %%a in ('powershell -NoProfile -Command "(Get-Date).AddDays(-1).ToString('yyyy-MM-dd')"') do set DATE=%%a-%%b-%%c
    set EXTRA= --llm --verbose
    goto run
)

set DATE=%~1
shift
set EXTRA=
:collect
if "%~1"=="" goto run
set EXTRA=%EXTRA% %~1
shift
goto collect

:run
echo Running report for %DATE%%EXTRA%
docker compose run --rm -e CRON_SCHEDULE= lkml-tracker --date %DATE%%EXTRA%
