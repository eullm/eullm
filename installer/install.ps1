<#
.SYNOPSIS
    Install, upgrade or uninstall EuLLM Engine on Windows.

.DESCRIPTION
    Quick install (PowerShell):

        irm https://raw.githubusercontent.com/eullm/eullm/main/installer/install.ps1 | iex

    Picks the CUDA build when an NVIDIA GPU the CUDA build covers (compute
    capability 8.6, 8.9 or 12.0) is present on driver 580+, and the CPU build
    otherwise, verifies the download against the release's checksums.txt,
    installs into a per-user directory and adds it to the user PATH. No
    administrator rights are needed.

    Environment variables (all optional):

        EULLM_VERSION      Release to install, e.g. 0.7.9 (default: latest stable)
        EULLM_INSTALL_DIR  Install directory (default: %LOCALAPPDATA%\Programs\EuLLM)
        EULLM_VARIANT      cpu or cuda, to skip GPU detection
        EULLM_UNINSTALL    Set to 1 to remove EuLLM and its PATH entry

    The Linux/macOS counterpart is installer/install.sh.

.EXAMPLE
    $env:EULLM_VARIANT = "cpu"; irm https://raw.githubusercontent.com/eullm/eullm/main/installer/install.ps1 | iex
#>

# Everything runs inside a function: this script is usually piped into
# `iex`, where a top-level `exit` would close the user's terminal and
# top-level variables would leak into their session.
function Install-EuLLM {
    [CmdletBinding()]
    param()

    $ErrorActionPreference = 'Stop'
    # Invoke-WebRequest's progress bar slows large downloads down by an
    # order of magnitude on Windows PowerShell 5.1.
    $ProgressPreference = 'SilentlyContinue'
    # Windows PowerShell 5.1 on older Windows 10 builds does not offer
    # TLS 1.2 by default, and GitHub refuses anything older.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $repo = 'eullm/eullm'
    $installDir = if ($env:EULLM_INSTALL_DIR) { $env:EULLM_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\EuLLM' }

    if ($env:EULLM_UNINSTALL -eq '1') {
        Uninstall-EuLLM -InstallDir $installDir
        return
    }

    if (-not [Environment]::Is64BitOperatingSystem) {
        throw 'EuLLM needs 64-bit Windows.'
    }
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        Write-Warning 'There is no native Windows ARM64 build yet; installing the x64 CPU build, which runs under emulation.'
    }

    $variant = $env:EULLM_VARIANT
    if (-not $variant) { $variant = Get-EuLLMVariant }
    # Candidates in order of preference; the first one the release lists
    # in checksums.txt is installed. The CPU ZIP carries the Visual C++
    # runtime next to the exe, so it runs on a Windows without the VC++
    # Redistributable; releases up to 0.7.9 only have the bare exe.
    switch ($variant) {
        'cpu'  { $candidates = @('eullm-windows-x64.zip', 'eullm-windows-x64.exe') }
        'cuda' { $candidates = @('eullm-windows-x64-cuda-13.1.zip') }
        default { throw "Unknown EULLM_VARIANT '$variant' (expected cpu or cuda)." }
    }

    $base = if ($env:EULLM_VERSION) {
        "https://github.com/$repo/releases/download/EuLLM-v$($env:EULLM_VERSION.TrimStart('v'))"
    } else {
        "https://github.com/$repo/releases/latest/download"
    }

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("eullm-install-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $sums = Join-Path $tmp 'checksums.txt'
        Invoke-WebRequest -UseBasicParsing -Uri "$base/checksums.txt" -OutFile $sums

        # checksums.txt lines look like "<hash>  <artifact-dir>/<file>", so
        # match on the file name at the end of the path.
        $listed = @{}
        foreach ($line in Get-Content $sums) {
            $parts = $line -split '\s+', 2
            if ($parts.Count -eq 2) { $listed[($parts[1] -split '/')[-1]] = $parts[0] }
        }
        $asset = $candidates | Where-Object { $listed.ContainsKey($_) } | Select-Object -First 1
        if (-not $asset) { throw "None of $($candidates -join ', ') is listed in checksums.txt, refusing to install an unverified binary." }
        $expected = $listed[$asset]

        Write-Host "Installing $asset (variant: $variant) into $installDir"
        if ($variant -eq 'cuda') { Write-Host 'The CUDA build is about 500 MB, this can take a while.' }
        $file = Join-Path $tmp $asset
        Invoke-WebRequest -UseBasicParsing -Uri "$base/$asset" -OutFile $file

        $actual = (Get-FileHash -Algorithm SHA256 -Path $file).Hash
        if ($actual -ne $expected) { throw "Checksum mismatch for $asset (expected $expected, got $actual)." }
        Write-Host 'Checksum OK'

        $running = Get-Process -Name eullm -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -and $_.Path.StartsWith($installDir, [StringComparison]::OrdinalIgnoreCase) }
        if ($running) { throw "EuLLM is running from $installDir (PID $($running.Id -join ', ')). Stop it and run the installer again." }

        New-Item -ItemType Directory -Path $installDir -Force | Out-Null
        # Remove what a previous install left behind, so switching from the
        # CUDA build to the CPU one does not leave stale DLLs around.
        Get-ChildItem -Path $installDir -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq 'eullm.exe' -or $_.Extension -eq '.dll' -or $_.Name -like 'THIRD-PARTY-NOTICES*' } |
            Remove-Item -Force

        if ($asset -like '*.zip') {
            $unzip = Join-Path $tmp 'unzipped'
            Expand-Archive -Path $file -DestinationPath $unzip
            Get-ChildItem -Path $unzip -File | Copy-Item -Destination $installDir
        } else {
            Copy-Item -Path $file -Destination (Join-Path $installDir 'eullm.exe')
        }
        Get-ChildItem -Path $installDir -File | Unblock-File

        Add-EuLLMToPath -Dir $installDir

        Write-Host ''
        Write-Host "EuLLM installed: $(Join-Path $installDir 'eullm.exe')"
        Write-Host ''
        Write-Host 'Try it (in this window, or any new one):'
        Write-Host '  eullm run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M'
    } finally {
        Remove-Item -Recurse -Force -Path $tmp -ErrorAction SilentlyContinue
    }
}

# cuda when nvidia-smi reports a GPU the CUDA 13.1 build can actually run,
# cpu otherwise.
function Get-EuLLMVariant {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return 'cpu' }
    try {
        # compute_cap needs a driver from 2021 on; older ones print an error
        # there, which fails the match below and lands in the cpu branch.
        $out = & $smi.Source --query-gpu=driver_version,compute_cap --format=csv,noheader 2>$null |
            Select-Object -First 1
    } catch {
        return 'cpu'
    }
    if ($out -notmatch '^\s*(\d+)\.\d+\s*,\s*(\d+\.\d+)\s*$') { return 'cpu' }
    $major = [int]$Matches[1]
    $cap = $Matches[2]
    # The Windows CUDA bundle is built for 8.6;89;120 with no PTX, so a card
    # outside that set cannot run it: the driver version says nothing about
    # the architecture, and the A100/H100 have no Windows build at all.
    if ($cap -notin '8.6', '8.9', '12.0') {
        Write-Warning "NVIDIA GPU with compute capability $cap is not covered by the CUDA build (8.6, 8.9, 12.0); installing the CPU build. Set `$env:EULLM_VARIANT='cuda' to install the CUDA build anyway."
        return 'cpu'
    }
    if ($major -lt 580) {
        Write-Warning "NVIDIA driver $major is older than 580, which the CUDA build needs; installing the CPU build. Update the driver and run the installer again for GPU support."
        return 'cpu'
    }
    return 'cuda'
}

function Add-EuLLMToPath {
    param([string]$Dir)
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @($userPath -split ';' | Where-Object { $_ })
    if ($entries -notcontains $Dir) {
        [Environment]::SetEnvironmentVariable('Path', (($entries + $Dir) -join ';'), 'User')
        Write-Host "Added $Dir to your user PATH"
    }
    # The registry change only reaches new processes; make `eullm` work in
    # this window too.
    if (@($env:Path -split ';') -notcontains $Dir) { $env:Path = "$env:Path;$Dir" }
}

function Uninstall-EuLLM {
    param([string]$InstallDir)
    if (Test-Path $InstallDir) {
        Remove-Item -Recurse -Force -Path $InstallDir
        Write-Host "Removed $InstallDir"
    } else {
        Write-Host "$InstallDir does not exist, nothing to remove"
    }
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @($userPath -split ';' | Where-Object { $_ -and $_ -ne $InstallDir })
    [Environment]::SetEnvironmentVariable('Path', ($entries -join ';'), 'User')
    Write-Host 'Removed EuLLM from your user PATH. Downloaded models and the audit log are kept in %USERPROFILE%\.eullm.'
}

Install-EuLLM
