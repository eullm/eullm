<#
.SYNOPSIS
    Install, upgrade or uninstall EuLLM Engine on Windows.

.DESCRIPTION
    Quick install (PowerShell):

        irm https://raw.githubusercontent.com/eullm/eullm/main/installer/install.ps1 | iex

    Picks the CUDA build when an NVIDIA GPU the CUDA build covers (compute
    capability 8.6, 8.9 or 12.0) is present on driver 580+; the Vulkan build
    for another GPU Vulkan serves (an AMD Radeon, an Intel Arc, an NVIDIA card
    outside the CUDA build) when the graphics driver installed the Vulkan
    loader; and the CPU build otherwise. It verifies the download against the
    release's checksums.txt, installs into a per-user directory and adds it to
    the user PATH. No administrator rights are needed.

    Environment variables (all optional):

        EULLM_VERSION      Release to install, e.g. 0.7.9 (default: latest stable)
        EULLM_INSTALL_DIR  Install directory (default: %LOCALAPPDATA%\Programs\EuLLM)
        EULLM_VARIANT      cpu, cuda or vulkan, to skip GPU detection
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
    $detected = -not $variant
    if ($detected) { $variant = Get-EuLLMVariant }
    # Candidates in order of preference; the first one the release lists
    # in checksums.txt is installed. The CPU ZIP carries the Visual C++
    # runtime next to the exe, so it runs on a Windows without the VC++
    # Redistributable; releases up to 0.7.9 only have the bare exe.
    $cpuCandidates = @('eullm-windows-x64.zip', 'eullm-windows-x64.exe')
    switch ($variant) {
        'cpu'    { $candidates = $cpuCandidates }
        'cuda'   { $candidates = @('eullm-windows-x64-cuda-13.1.zip') }
        'vulkan' { $candidates = @('eullm-windows-x64-vulkan.zip') }
        # ROCm's own layout (bin\ and .kpack\), which this script, copying a
        # ZIP's top-level files, cannot install yet.
        'rocm'   { throw "The Windows ROCm build is not installed by this script yet: download eullm-windows-x64-rocm.zip from https://github.com/eullm/eullm/releases/latest, extract it, and run bin\eullm.exe from there." }
        default { throw "Unknown EULLM_VARIANT '$variant' (expected cpu, cuda or vulkan)." }
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

        # checksums.txt lines look like "<hash>  <file>", but match on the file
        # name at the end of the path: releases up to 0.7.20 listed the
        # download artifact directory in front of it ("<hash>  <dir>/<file>")
        # and this has to keep reading those too.
        $listed = @{}
        foreach ($line in Get-Content $sums -Encoding UTF8) {
            $parts = $line -split '\s+', 2
            if ($parts.Count -eq 2) { $listed[($parts[1] -split '/')[-1]] = $parts[0] }
        }
        $asset = $candidates | Where-Object { $listed.ContainsKey($_) } | Select-Object -First 1
        # A detected GPU build the release does not carry: releases before
        # the Vulkan one have no Vulkan ZIP, and a release whose GPU build
        # failed publishes without it. The CPU build still runs; a variant
        # asked for by name still fails below, since it was asked for.
        if (-not $asset -and $detected -and $variant -ne 'cpu') {
            Write-Warning "This release has no $variant build for Windows; installing the CPU build."
            $variant = 'cpu'
            $candidates = $cpuCandidates
            $asset = $candidates | Where-Object { $listed.ContainsKey($_) } | Select-Object -First 1
        }
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

# What Select-EuLLMVariant decides from, read off this machine: nvidia-smi's
# driver version and compute capability, the display adapters' names, and
# whether the graphics driver installed the Vulkan loader.
function Get-EuLLMVariant {
    $smiLine = ''
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            # compute_cap needs a driver from 2021 on; older ones print an
            # error there, which Select-EuLLMVariant does not take for one.
            $smiLine = [string](& $smi.Source --query-gpu=driver_version,compute_cap --format=csv,noheader 2>$null |
                Select-Object -First 1)
        } catch {
            $smiLine = ''
        }
    }
    $gpus = @()
    try {
        $gpus = @(Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop | ForEach-Object { $_.Name })
    } catch {
        $gpus = @()
    }
    # Every AMD, Intel and NVIDIA driver installs vulkan-1.dll here, and
    # without it the Vulkan build cannot start at all, so it is the test for
    # a driver that serves Vulkan.
    $hasVulkan = Test-Path -LiteralPath "$([Environment]::SystemDirectory)\vulkan-1.dll"
    Select-EuLLMVariant -Smi $smiLine -Gpus $gpus -HasVulkan $hasVulkan
}

# cuda when nvidia-smi reports a GPU the CUDA 13.1 build can run; vulkan for
# a GPU the Vulkan build serves, when the Vulkan loader is installed; cpu
# otherwise. Kept apart from the queries above so CI can test it without the
# hardware.
function Select-EuLLMVariant {
    param(
        [string]$Smi,
        [string[]]$Gpus = @(),
        [bool]$HasVulkan
    )
    # AMD and NVIDIA adapters of any kind, and Intel's Arc (the cards, and the
    # integrated graphics of Core Ultra chips, which carry the same name).
    # Older Intel integrated graphics (UHD, Iris Xe) stay on the CPU build,
    # which is rarely slower there and has no driver of its own to go wrong.
    $vulkanGpu = $null
    if ($HasVulkan) {
        $vulkanGpu = $Gpus | Where-Object { $_ -match 'AMD|Radeon|NVIDIA|GeForce|Intel.*\bArc\b' } |
            Select-Object -First 1
    }
    $other = if ($vulkanGpu) { 'vulkan' } else { 'cpu' }
    $otherName = if ($vulkanGpu) { 'Vulkan' } else { 'CPU' }

    if ($Smi -match '^\s*(\d+)\.\d+\s*,\s*(\d+\.\d+)\s*$') {
        $major = [int]$Matches[1]
        $cap = $Matches[2]
        # The Windows CUDA bundle is built for 8.6;89;120 with no PTX, so a
        # card outside that set cannot run it: the driver version says nothing
        # about the architecture, and the A100/H100 have no Windows build at
        # all. The Vulkan build serves the consumer cards among them (GTX
        # 1000, RTX 2000) and drivers older than 580.
        if ($cap -notin '8.6', '8.9', '12.0') {
            Write-Warning "NVIDIA GPU with compute capability $cap is not covered by the CUDA build (8.6, 8.9, 12.0); installing the $otherName build. Set `$env:EULLM_VARIANT='cuda' to install the CUDA build anyway."
            return $other
        }
        if ($major -lt 580) {
            Write-Warning "NVIDIA driver $major is older than 580, which the CUDA build needs; installing the $otherName build. Update the driver and run the installer again for the CUDA build."
            return $other
        }
        return 'cuda'
    }
    if ($vulkanGpu) { Write-Host "GPU: $vulkanGpu, installing the Vulkan build" }
    return $other
}

function Set-EuLLMUserPath {
    param([string[]]$Entries)
    # REG_EXPAND_SZ, which is what Windows itself keeps a user PATH as, and
    # what [Environment]::SetEnvironmentVariable does NOT write: it creates a
    # plain REG_SZ, keeping the text but losing the expansion. On a machine
    # whose user PATH holds %USERPROFILE%\bin, %GOPATH%\bin or
    # %LOCALAPPDATA%\Programs\..., installing EuLLM silently froze those
    # entries, and from the next logon the composed environment contains
    # directories literally named "%USERPROFILE%\bin" — so pip, go and java
    # break for that user, with nothing in the installer's output to show for
    # it. Writing through the registry keeps the type the value already had.
    Set-ItemProperty -Path HKCU:\Environment -Name Path -Type ExpandString `
        -Value ($Entries -join ';')
}

function Add-EuLLMToPath {
    param([string]$Dir)
    # The raw text, %VAR% entries as written: GetEnvironmentVariable returns
    # them expanded, and writing that back would freeze them.
    $userPath = (Get-Item HKCU:\Environment).GetValue('Path', '', 'DoNotExpandEnvironmentNames')
    $entries = @($userPath -split ';' | Where-Object { $_ })
    if ($entries -notcontains $Dir) {
        Set-EuLLMUserPath -Entries ($entries + $Dir)
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
    $userPath = (Get-Item HKCU:\Environment).GetValue('Path', '', 'DoNotExpandEnvironmentNames')
    $entries = @($userPath -split ';' | Where-Object { $_ -and $_ -ne $InstallDir })
    # Only when there was something to remove: an uninstall on a machine that
    # never had EuLLM should not rewrite the user's PATH at all.
    if ($entries.Count -lt @($userPath -split ';' | Where-Object { $_ }).Count) {
        Set-EuLLMUserPath -Entries $entries
        Write-Host 'Removed EuLLM from your user PATH. Downloaded models and the audit log are kept in %USERPROFILE%\.eullm.'
    } else {
        Write-Host 'EuLLM was not in your user PATH, left it alone.'
    }
}

Install-EuLLM
