@echo off
setlocal

echo [1/3] Stopping containers...
docker compose down
if %ERRORLEVEL% neq 0 (
    echo ERROR: docker compose down failed.
    exit /b %ERRORLEVEL%
)

echo [2/3] Building image...
docker compose build
if %ERRORLEVEL% neq 0 (
    echo ERROR: docker compose build failed.
    exit /b %ERRORLEVEL%
)

echo [3/3] Starting containers...
docker compose up -d
if %ERRORLEVEL% neq 0 (
    echo ERROR: docker compose up failed.
    exit /b %ERRORLEVEL%
)

echo.
echo Done. Containers are running.
docker compose ps

endlocal
