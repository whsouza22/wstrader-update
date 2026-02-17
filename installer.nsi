; ================================
; WS Trader AI - NSIS Installer Script
; ================================

!include "MUI2.nsh"
!include "FileFunc.nsh"

; ================================
; Configurações Gerais
; ================================
Name "WS Trader AI"
OutFile "WsTrader_Setup_2.5.exe"
InstallDir "$PROGRAMFILES64\WsTrader"
InstallDirRegKey HKLM "Software\WsTrader" "InstallDir"
RequestExecutionLevel admin

; ================================
; Interface do Instalador
; ================================
!define MUI_ICON "Img\ws_ai_trader_corrigido.ico"
!define MUI_UNICON "Img\ws_ai_trader_corrigido.ico"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Header\orange.bmp"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Wizard\orange.bmp"

!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\WsTrader.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Executar WS Trader AI"

; ================================
; Páginas do Instalador (SEM licença)
; ================================
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; ================================
; Páginas do Desinstalador
; ================================
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; ================================
; Idiomas
; ================================
!insertmacro MUI_LANGUAGE "PortugueseBR"

; ================================
; Informações da Versão
; ================================
VIProductVersion "2.5.0.0"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "ProductName" "WS Trader AI"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "CompanyName" "WS Trader Team"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "LegalCopyright" "© 2026 WS Trader Team"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "FileDescription" "Assistente Inteligente de Trading"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "FileVersion" "2.5"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "ProductVersion" "2.5"

; ================================
; Seção Principal - Instalação
; ================================
Section "WS Trader AI" SecMain
    SectionIn RO ; Obrigatório

    ; Fecha processos antes de instalar
    nsExec::ExecToLog 'taskkill /F /IM WsTrader.exe /T'
    Sleep 2000

    ; Define o diretório de saída
    SetOutPath "$INSTDIR"
    
    ; Remove instalação antiga para garantir atualização limpa
    RMDir /r "$INSTDIR\_internal"
    Delete "$INSTDIR\WsTrader.exe"
    
    ; Força sobrescrever SEMPRE (ignora datas/versões)
    SetOverwrite on
    SetOverwrite ifnewer
    SetOverwrite try

    ; Copia todos os arquivos da pasta dist\WsTrader
    File /r "dist\WsTrader\*.*"

    ; Cria atalhos
    CreateDirectory "$SMPROGRAMS\WS Trader AI"
    CreateShortcut "$SMPROGRAMS\WS Trader AI\WS Trader AI.lnk" "$INSTDIR\WsTrader.exe" "" "$INSTDIR\WsTrader.exe" 0
    CreateShortcut "$SMPROGRAMS\WS Trader AI\Desinstalar.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0

    ; Atalho na área de trabalho
    CreateShortcut "$DESKTOP\WS Trader AI.lnk" "$INSTDIR\WsTrader.exe" "" "$INSTDIR\WsTrader.exe" 0

    ; Salva informações no registro
    WriteRegStr HKLM "Software\WsTrader" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "Software\WsTrader" "Version" "2.5"

    ; Cria desinstalador
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Adiciona no Adicionar/Remover Programas
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "DisplayName" "WS Trader AI"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "DisplayIcon" "$INSTDIR\WsTrader.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "Publisher" "WS Trader Team"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "DisplayVersion" "2.5"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "NoRepair" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "EstimatedSize" 320000

SectionEnd

; ================================
; Seção de Desinstalação
; ================================
Section "Uninstall"
    ; Remove arquivos
    RMDir /r "$INSTDIR"

    ; Remove atalhos
    Delete "$SMPROGRAMS\WS Trader AI\*.*"
    RMDir "$SMPROGRAMS\WS Trader AI"
    Delete "$DESKTOP\WS Trader AI.lnk"

    ; Remove entradas do registro
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader"
    DeleteRegKey HKLM "Software\WsTrader"

SectionEnd

; ================================
; Funções Auxiliares
; ================================
Function .onInit
    ; Fecha o aplicativo se estiver rodando (em qualquer modo)
    nsExec::ExecToLog 'taskkill /F /IM WsTrader.exe'
    Sleep 2000
    
    ; Verifica modo silencioso (para auto-update)
    ${GetParameters} $R1
    ${GetOptions} $R1 "/S" $R2
    IfErrors 0 silent_mode

    ; Modo normal - verifica se já está instalado
    ReadRegStr $R0 HKLM "Software\WsTrader" "InstallDir"
    StrCmp $R0 "" done

    MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION \
    "WS Trader AI ja esta instalado em $R0$\n$\nClique em 'OK' para desinstalar a versao anterior.$\nClique em 'Cancel' para cancelar." \
    IDOK uninst
    Abort

    uninst:
        ExecWait '$R0\Uninstall.exe /S _?=$R0'
        Delete "$R0\Uninstall.exe"
        RMDir $R0
        Goto done

    silent_mode:
        ; Modo silencioso - apenas sobrescreve os arquivos
        ReadRegStr $R0 HKLM "Software\WsTrader" "InstallDir"
        StrCmp $R0 "" done
    done:
FunctionEnd

Function un.onInit
    ; Verifica modo silencioso
    ${GetParameters} $R1
    ${GetOptions} $R1 "/S" $R2
    IfErrors 0 +3
    MessageBox MB_YESNO "Tem certeza que deseja desinstalar o WS Trader AI?" IDYES +2
    Abort
FunctionEnd
