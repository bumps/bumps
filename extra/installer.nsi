;-------------------------------------------------------------------------------
; Includes
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "WinVer.nsh"
!include "x64.nsh"

;-------------------------------------------------------------------------------
; Constants
!ifndef PRODUCT_NAME
    !define PRODUCT_NAME "Bumps"
!endif
!define PRODUCT_DESCRIPTION "Bayesian uncertainty modeling for the physical sciences"
!define COPYRIGHT "Copyright © 2024 The Bumps developers"
!define PRODUCT_VERSION 1.0.0.0
!define SETUP_VERSION 1.0.0.0
!ifndef SRC
    !define SRC "..\conda_packed"
!endif

;-------------------------------------------------------------------------------
; Attributes
Name "${PRODUCT_NAME}"
OutFile "BumpsWebviewSetup.exe"
InstallDir "$LocalAppData\${PRODUCT_NAME}"
InstallDirRegKey HKCU "Software\Reflectometry-Org\${PRODUCT_NAME}" ""
RequestExecutionLevel user ; user|highest|admin

;-------------------------------------------------------------------------------
; Version Info
VIProductVersion "${PRODUCT_VERSION}"
VIAddVersionKey "ProductName" "${PRODUCT_NAME}"
VIAddVersionKey "ProductVersion" "${PRODUCT_VERSION}"
VIAddVersionKey "FileDescription" "${PRODUCT_DESCRIPTION}"
VIAddVersionKey "LegalCopyright" "${COPYRIGHT}"
VIAddVersionKey "FileVersion" "${SETUP_VERSION}"

;-------------------------------------------------------------------------------
; Modern UI Appearance
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\orange-install.ico"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Header\orange.bmp"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Wizard\orange.bmp"
!define MUI_FINISHPAGE_NOAUTOCLOSE

;-------------------------------------------------------------------------------
; Installer Pages
!insertmacro MUI_PAGE_WELCOME
;!insertmacro MUI_PAGE_LICENSE "${NSISDIR}\Docs\Modern UI\License.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

;-------------------------------------------------------------------------------
; Uninstaller Pages
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

;-------------------------------------------------------------------------------
; Languages
!insertmacro MUI_LANGUAGE "English"

;-------------------------------------------------------------------------------
; Installer Sections
Section "Webview Server" SEC01
    SetOutPath "$INSTDIR"
    File /r "${SRC}\*"
    WriteRegStr HKCU "Software\Reflectometry-Org\${PRODUCT_NAME}" "Install_Dir" "$INSTDIR"
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Registry entries
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                     "DisplayName" "${PRODUCT_NAME}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                     "UninstallString" '"$INSTDIR\uninstall.exe"'
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                      "NoModify" 1
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                      "NoRepair" 1
SectionEnd

Section "Start Menu Shortcuts" SEC02
    CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\BumpsWebview.lnk" \
        "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" \
        '-Command ""$INSTDIR\python.exe -m bumps.webview.server""' \
        "$INSTDIR\share\icons\bumps.ico"
    SetOutPath "%USERPROFILE%"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\BumpsPowershell.lnk" \
        "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" \
        '-NoExit -NoProfile -Command ""$INSTDIR\Library\bin\micromamba.exe shell hook -s powershell | Out-String | Invoke-Expression ; micromamba activate $INSTDIR""' \
        ""

SectionEnd

;-------------------------------------------------------------------------------
; Uninstaller Sections
Section "Uninstall"
	Delete "$INSTDIR\Uninstall.exe"
    RMDir /r /REBOOTOK "$INSTDIR"
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\*.lnk"
    RMDir  "$SMPROGRAMS\${PRODUCT_NAME}"
	DeleteRegKey /ifempty HKCU "Software\Reflectometry-Org\${PRODUCT_NAME}"
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
SectionEnd
