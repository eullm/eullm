<#
.SYNOPSIS
    Build the Microsoft Store MSIX for EuLLM from the two Windows builds.

.DESCRIPTION
    Lays out installer/msix/AppxManifest.xml and its Assets next to the CPU
    and CUDA builds (cpu\ and cuda\, each with its DLLs) and packs them with
    MakeAppx from the runner's Windows SDK. MakeAppx validates the manifest
    against the schema while packing, so a malformed manifest fails here
    rather than at Store certification.

    The package is left unsigned: the Microsoft Store signs it after
    certification. To install it locally before submitting, see
    installer/msix/README.md.

.PARAMETER CpuDir
    Folder holding the CPU eullm.exe and its DLLs (the extracted
    eullm-windows-x64.zip).

.PARAMETER CudaDir
    Folder holding the CUDA eullm.exe and its DLLs (the extracted
    eullm-windows-x64-cuda-13.1.zip).

.PARAMETER Version
    Package version, four numeric parts with the last one 0 (e.g. 0.7.9.0).

.PARAMETER Out
    Path of the .msix to write.
#>
param(
    [Parameter(Mandatory)] [string]$CpuDir,
    [Parameter(Mandatory)] [string]$CudaDir,
    [Parameter(Mandatory)] [string]$Version,
    [Parameter(Mandatory)] [string]$Out
)

$ErrorActionPreference = 'Stop'

if ($Version -notmatch '^\d+\.\d+\.\d+\.0$') {
    throw "MSIX version '$Version' must be four numbers ending in .0 (the Store reserves the fourth part)"
}
foreach ($dir in $CpuDir, $CudaDir) {
    if (-not (Test-Path (Join-Path $dir 'eullm.exe'))) { throw "no eullm.exe in $dir" }
}

$src = Join-Path $PSScriptRoot '..\..\installer\msix'
$layout = Join-Path ([IO.Path]::GetTempPath()) ("eullm-msix-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $layout | Out-Null
try {
    Copy-Item -Recurse (Join-Path $src 'Assets') $layout
    New-Item -ItemType Directory -Path "$layout\cpu", "$layout\cuda" | Out-Null
    Copy-Item "$CpuDir\*" "$layout\cpu" -Recurse
    Copy-Item "$CudaDir\*" "$layout\cuda" -Recurse

    # Set the version on the Identity element only, through the XML DOM
    # rather than a text replace, so nothing else in the manifest can match.
    [xml]$manifest = Get-Content (Join-Path $src 'AppxManifest.xml') -Raw -Encoding UTF8
    $manifest.Package.Identity.Version = $Version
    $manifest.Save((Join-Path $layout 'AppxManifest.xml'))

    $makeappx = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\makeappx.exe" |
        Where-Object { $_.Directory.Parent.Name -match '^\d+(\.\d+)+$' } |
        Sort-Object { [version]($_.Directory.Parent.Name) } -Descending |
        Select-Object -First 1
    if (-not $makeappx) { throw 'makeappx.exe not found in the Windows SDK' }
    Write-Host "Packing with $($makeappx.FullName)"

    & $makeappx.FullName pack /d $layout /p $Out /o
    if ($LASTEXITCODE -ne 0) { throw "makeappx failed with exit code $LASTEXITCODE" }
    Write-Host "Wrote $Out ($([math]::Round((Get-Item $Out).Length / 1MB)) MB), version $Version"
} finally {
    Remove-Item -Recurse -Force $layout -ErrorAction SilentlyContinue
}
