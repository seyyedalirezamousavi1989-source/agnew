#define AppName "LocalChat"
#define AppVersion "2.0.5"
#define Publisher "LocalChat"
#ifndef PayloadDir
  #define PayloadDir "..\build\payload\LocalChat"
#endif

[Setup]
AppId={{B7C2D5A1-6E3F-4D1B-9A52-0A1C2D3E4F50}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
AppPublisherURL=https://ollama.com/
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
OutputDir=..\build\installer
OutputBaseFilename=LocalChat-Setup-{#AppVersion}

; The application payload is bundled in this one Setup.exe. Only OS-level
; prerequisites (WebView2 and VC++ runtime) are downloaded when missing.
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\LocalChat.exe
WizardStyle=modern
SetupIconFile=icon.ico
LicenseFile=INSTALLER_LICENSE_FA.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "persian"; MessagesFile: "Farsi.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked
Name: "startup"; Description: "Run LocalChat when I sign in to Windows"; GroupDescription: "Optional startup:"; Flags: unchecked

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\LocalChat.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\LocalChat.exe"; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\LocalChat.exe"; Tasks: startup

[Run]
Filename: "{app}\LocalChat.exe"; Description: "Launch LocalChat now"; Flags: nowait postinstall skipifsilent

[Code]
var IntroPage: TWizardPage;
    IntroText: TNewStaticText;

function IsWebView2Installed: Boolean;
var Version: String;
begin
  Result := False;
  if RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) then
    Result := (Version <> '') and (Version <> '0.0.0.0');
  if not Result and RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) then
    Result := (Version <> '') and (Version <> '0.0.0.0');
  if not Result and RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) then
    Result := (Version <> '') and (Version <> '0.0.0.0');
end;

function IsVCRedistInstalled: Boolean;
var Version: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Version) and (Version = 1);
end;

function DownloadAndRun(const Url, FileName, Params, DisplayName: String): Boolean;
var TempFile: String;
  ResultCode: Integer;
begin
  Result := False;
  try
    WizardForm.StatusLabel.Caption := 'Downloading ' + DisplayName + '...';
    DownloadTemporaryFile(Url, FileName, '', nil);
    TempFile := ExpandConstant('{tmp}\') + FileName;
    WizardForm.StatusLabel.Caption := 'Installing ' + DisplayName + '...';
    if not ShellExec('open', TempFile, Params, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      RaiseException('Could not start ' + DisplayName + ' installer.');
    Result := (ResultCode = 0) or (ResultCode = 3010);
  except
    MsgBox('Could not download or install ' + DisplayName + '.' + #13#10 +
      'Check your internet connection and run Setup again.' + #13#10 + GetExceptionMessage, mbError, MB_OK);
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not IsVCRedistInstalled then begin
    if not DownloadAndRun('https://aka.ms/vc14/vc_redist.x64.exe', 'vc_redist.x64.exe', '/install /quiet /norestart', 'Microsoft Visual C++ Runtime') then
      Result := 'Microsoft Visual C++ Runtime installation failed.';
  end;
  if (Result = '') and not IsWebView2Installed then begin
    if not DownloadAndRun('https://go.microsoft.com/fwlink/p/?LinkId=2124703', 'MicrosoftEdgeWebView2RuntimeInstallerX64.exe', '/silent /install', 'Microsoft Edge WebView2 Runtime') then
      Result := 'Microsoft Edge WebView2 Runtime installation failed.';
  end;
end;

procedure InitializeWizard;
begin
  IntroPage := CreateCustomPage(wpWelcome, 'About LocalChat', 'A private local-first AI chat application');
  IntroText := TNewStaticText.Create(IntroPage);
  IntroText.Parent := IntroPage.Surface;
  IntroText.Left := ScaleX(0);
  IntroText.Top := ScaleY(8);
  IntroText.Width := IntroPage.SurfaceWidth;
  IntroText.Height := ScaleY(150);
  IntroText.AutoSize := False;
  IntroText.WordWrap := True;
  IntroText.Caption :=
    'LocalChat brings local Ollama models, optional online services, file attachments, memory and voice features into one desktop application.' + #13#10 + #13#10 +
    'The installer lets you choose the installation folder. Python is NOT required on this computer: the application runtime is included in this Setup file. If required, Setup downloads and installs Microsoft Visual C++ Runtime and Microsoft Edge WebView2. Ollama, AI models, speech engines and optional providers are downloaded from inside LocalChat after installation.' + #13#10 + #13#10 +
    'Your conversations and settings are stored separately in your Windows user profile and are preserved when you uninstall the application.';
end;

function InitializeUninstall: Boolean;
begin
  Result := True;
  if MsgBox('Your LocalChat data, conversations, settings and downloaded models will be kept. Continue uninstalling the application?', mbConfirmation, MB_YESNO) = IDNO then
    Result := False;
end;
