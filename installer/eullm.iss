; ─── EuLLM Desktop Installer ────────────────────────────────────────────────
;
; Inno Setup script that produces a single one-click Windows installer for
; EuLLM Engine, including the embedded chat UI. Three variants, all built
; from this same script using /D defines from CI:
;
;   ISCC.exe /DMyVariant=CPU            /DMyAppVersion=0.5.2 /DMyStagingDir=staging\cpu  eullm.iss
;   ISCC.exe /DMyVariant=CUDA           /DMyAppVersion=0.5.2 /DMyStagingDir=staging\cuda eullm.iss
;   ISCC.exe /DMyVariant=CUDA-TurboQuant /DMyAppVersion=0.5.2 /DMyStagingDir=staging\tq  eullm.iss
;
; Each staging dir must contain eullm.exe (+ the CUDA DLLs for the GPU variants).
; The chat UI is already baked into eullm.exe — no extra files needed.
;
; Design choices:
;   - Per-user install (PrivilegesRequired=lowest) -> no UAC prompt, no admin needed
;   - Default install dir under %LOCALAPPDATA%\Programs\EuLLM (no admin scope)
;   - PATH modification is OPT-IN (some users prefer not to pollute PATH)
;   - Start Menu shortcuts: "EuLLM Chat" launches the engine + opens browser
;   - Uninstaller registered with Programs & Features
; ───────────────────────────────────────────────────────────────────────────

#ifndef MyAppName
  #define MyAppName       "EuLLM Engine"
#endif

#ifndef MyAppVersion
  #define MyAppVersion    "0.5.2"
#endif

#ifndef MyVariant
  #define MyVariant       "CPU"
#endif

#ifndef MyStagingDir
  #define MyStagingDir    "staging"
#endif

#ifndef MyAppPublisher
  #define MyAppPublisher  "I3K Technologies"
#endif

#ifndef MyAppURL
  #define MyAppURL        "https://eullm.eu"
#endif

#define MyAppExeName       "eullm.exe"
#define MyAppLauncherName  "eullm-chat.ps1"
#define MyOutputBaseName   "EuLLM-Setup-" + MyVariant + "-" + MyAppVersion

[Setup]
AppId={{6B7C5C2A-EU11-44C7-9F0F-EU11M0CHAT01}}
AppName={#MyAppName} ({#MyVariant})
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyVariant} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL=https://github.com/eullm/eullm
AppUpdatesURL=https://github.com/eullm/eullm/releases
DefaultDirName={localappdata}\Programs\EuLLM
DefaultGroupName=EuLLM
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
OutputDir=output
OutputBaseFilename={#MyOutputBaseName}
SetupIconFile=eullm.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
DisableDirPage=no
DisableReadyPage=no
ShowLanguageDialog=no
LicenseFile=..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktopicon";    Description: "Create a desktop shortcut for EuLLM Chat"; GroupDescription: "Additional shortcuts:"
Name: "modifypath";     Description: "Add EuLLM to my user PATH (so 'eullm' works from any terminal)"; GroupDescription: "Environment:"; Flags: unchecked

[Files]
; Main engine binary (always present)
Source: "{#MyStagingDir}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
; CUDA runtime DLLs (only present in CUDA variants — wildcards make this safe
; for the CPU build too: 0 matches is allowed with skipifsourcedoesntexist).
Source: "{#MyStagingDir}\cudart64_*.dll";   DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#MyStagingDir}\cublas64_*.dll";   DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#MyStagingDir}\cublasLt64_*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; Launcher script that opens browser to the chat
Source: "eullm-chat.ps1"; DestDir: "{app}"; Flags: ignoreversion
; License & README
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "README-installer.txt"; DestDir: "{app}"; DestName: "README.txt"; Flags: ignoreversion

[Icons]
; Start Menu
Name: "{userprograms}\EuLLM\EuLLM Chat";        Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\{#MyAppLauncherName}"""; IconFilename: "{app}\eullm.ico"; WorkingDir: "{app}"
Name: "{userprograms}\EuLLM\EuLLM CLI";         Filename: "{cmd}";          Parameters: "/k cd /d ""{userdocs}"""; IconFilename: "{app}\eullm.ico"; WorkingDir: "{userdocs}"; Comment: "Open a terminal in your Documents folder with 'eullm' on PATH"
Name: "{userprograms}\EuLLM\Uninstall EuLLM";   Filename: "{uninstallexe}"
; Desktop (opt-in)
Name: "{userdesktop}\EuLLM Chat";              Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\{#MyAppLauncherName}"""; IconFilename: "{app}\eullm.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Offer to launch the chat right after install (user can untick if just installing).
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\{#MyAppLauncherName}"""; \
    Description: "Launch EuLLM Chat now"; \
    Flags: postinstall nowait skipifsilent unchecked

[Code]
const
  EnvironmentKey = 'Environment';

procedure EnvAddPath(Path: string);
var
  Paths: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Paths) then
    Paths := '';
  if Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(Paths) + ';') > 0 then
    exit;
  Paths := Paths + ';' + Path;
  while (Length(Paths) > 0) and (Paths[1] = ';') do
    Delete(Paths, 1, 1);
  RegWriteStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Paths);
end;

procedure EnvRemovePath(Path: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Paths) then
    exit;
  P := Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(Paths) + ';');
  if P = 0 then
    exit;
  Delete(Paths, P - 1, Length(Path) + 1);
  RegWriteStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Paths);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and IsTaskSelected('modifypath') then
    EnvAddPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    EnvRemovePath(ExpandConstant('{app}'));
end;
