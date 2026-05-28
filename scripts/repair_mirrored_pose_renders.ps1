<#
.SYNOPSIS
Flip mirrored render companion images reported by audit_pose_image_alignment.py.

.DESCRIPTION
Reads pose_image_alignment_audit.csv and processes rows classified as
mirror_candidate. By default this is a dry run. Use -Apply to modify files.

The audit compares JSON/bone orientation against the rendered images. For a
mirror_candidate, the rendered companions are mirrored relative to the
bone_structure/OpenPose JSON, so the default action flips depth, normal, and
lineart images horizontally.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\repair_mirrored_pose_renders.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\repair_mirrored_pose_renders.ps1 -Apply -Backup

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\repair_mirrored_pose_renders.ps1 -Classification ambiguous -BaseNamePattern "*dance_03" -Apply -Backup
#>

[CmdletBinding()]
param(
    [string]$Report = "",
    [string]$Root = "",
    [ValidateSet("depth", "normal", "lineart", "bone_structure")]
    [string[]]$Kinds = @("depth", "normal", "lineart"),
    [string[]]$Classification = @("mirror_candidate"),
    [string[]]$BaseNamePattern = @("*"),
    [double]$MinFlipDelta = -1.0,
    [double]$MinFlippedScore = -1.0,
    [switch]$Apply,
    [switch]$Backup,
    [string]$Magick = "magick"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Report)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $Report = Join-Path $scriptRoot "..\pose_image_alignment_audit.csv"
}

if ([string]::IsNullOrWhiteSpace($Root)) {
    if (-not [string]::IsNullOrWhiteSpace($env:OPENPOSE_MODELS_PATH)) {
        $Root = $env:OPENPOSE_MODELS_PATH
    }
    else {
        $Root = "C:\EasyDiffusion\stable-diffusion\stable-diffusion-webui\models\openpose"
    }
}

function Resolve-AssetPath {
    param(
        [AllowEmptyString()][string]$Value,
        [Parameter(Mandatory = $true)][string]$RootPath
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }

    $relative = $Value -replace "/", "\"
    return [System.IO.Path]::GetFullPath((Join-Path $RootPath $relative))
}

function Test-AnyLike {
    param(
        [AllowEmptyString()][string]$Value,
        [Parameter(Mandatory = $true)][string[]]$Patterns
    )

    foreach ($pattern in $Patterns) {
        if ($Value -like $pattern) {
            return $true
        }
    }
    return $false
}

function Read-Score {
    param([AllowEmptyString()][string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }
    return [double]::Parse($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$reportPath = (Resolve-Path -LiteralPath $Report).Path
$magickCommand = Get-Command $Magick -ErrorAction Stop
$rows = Import-Csv -LiteralPath $reportPath

$targets = [ordered]@{}
foreach ($row in $rows) {
    if ($Classification -notcontains $row.classification) {
        continue
    }

    if (-not (Test-AnyLike -Value $row.base_name -Patterns $BaseNamePattern)) {
        continue
    }

    $flipDelta = Read-Score -Value $row.flip_delta
    $flippedScore = Read-Score -Value $row.flipped_score
    if ($null -ne $flipDelta -and $flipDelta -lt $MinFlipDelta) {
        continue
    }
    if ($null -ne $flippedScore -and $flippedScore -lt $MinFlippedScore) {
        continue
    }

    foreach ($kind in $Kinds) {
        $value = $row.$kind
        $path = Resolve-AssetPath -Value $value -RootPath $rootPath
        if ($null -eq $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) {
            continue
        }

        $resolved = (Resolve-Path -LiteralPath $path).Path
        $rootPrefix = $rootPath.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
        $isInsideRoot = $resolved.Equals($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
            $resolved.StartsWith(
                "$rootPrefix$([System.IO.Path]::DirectorySeparatorChar)",
                [System.StringComparison]::OrdinalIgnoreCase
            )
        if (-not $isInsideRoot) {
            throw "Refusing to process path outside root: $resolved"
        }

        $targets[$resolved] = $true
    }
}

Write-Host "Report: $reportPath"
Write-Host "Root:   $rootPath"
Write-Host "Mode:   $(if ($Apply) { 'APPLY' } else { 'DRY RUN' })"
Write-Host "Kinds:  $($Kinds -join ', ')"
Write-Host "Class:  $($Classification -join ', ')"
Write-Host "Names:  $($BaseNamePattern -join ', ')"
Write-Host "Scores: MinFlipDelta=$MinFlipDelta MinFlippedScore=$MinFlippedScore"
Write-Host "Files:  $($targets.Count)"

foreach ($path in $targets.Keys) {
    if ($Apply) {
        if ($Backup) {
            $backupPath = "$path.preflip.bak"
            if (-not (Test-Path -LiteralPath $backupPath)) {
                Copy-Item -LiteralPath $path -Destination $backupPath -ErrorAction Stop
            }
        }
        & $magickCommand.Source mogrify -flop $path
        if ($LASTEXITCODE -ne 0) {
            throw "ImageMagick failed for $path"
        }
        Write-Host "flipped $path"
    }
    else {
        Write-Host "would flip $path"
    }
}

if (-not $Apply) {
    Write-Host ""
    Write-Host "Dry run only. Add -Apply to write changes; add -Backup to create *.preflip.bak files first."
}
