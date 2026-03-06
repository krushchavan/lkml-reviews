# rebuild.ps1 - Stop, rebuild, and restart lkml-tracker Docker services

param(
    [switch]$NoCache,
    [switch]$All
)

$ErrorActionPreference = "Stop"

$projectDir = $PSScriptRoot
Set-Location $projectDir

Write-Host "Stopping containers..." -ForegroundColor Yellow
docker compose down

Write-Host "Rebuilding lkml-tracker image..." -ForegroundColor Yellow
if ($NoCache) {
    if ($All) {
        docker compose build --no-cache
    } else {
        docker compose build --no-cache lkml-tracker
    }
} else {
    if ($All) {
        docker compose build
    } else {
        docker compose build lkml-tracker
    }
}

Write-Host "Starting containers..." -ForegroundColor Yellow
docker compose up -d

Write-Host "Done. Container status:" -ForegroundColor Green
docker compose ps
