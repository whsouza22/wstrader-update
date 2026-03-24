; ================================
; WS Trader AI - NSIS Installer Script
; ================================

!include "MUI2.nsh"
!include "FileFunc.nsh"

; ================================
; Configurações Gerais
; ================================
Name "WS Trader AI"
OutFile "WsTrader_Setup_6.9.0.exe"
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
VIProductVersion "6.9.0.0"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "ProductName" "WS Trader AI"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "CompanyName" "WS Trader Team"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "LegalCopyright" "© 2026 WS Trader Team"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "FileDescription" "Assistente Inteligente de Trading"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "FileVersion" "6.9.0"
VIAddVersionKey /LANG=${LANG_PORTUGUESEBR} "ProductVersion" "6.9.0"

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

    ; ================================
    ; LIMPEZA TOTAL de caches e dados operacionais antigos
    ; Preserva APENAS: .env (credenciais) e preferences.json
    ; ================================

    ; --- ~/.wstrader/ (principal) ---
    Delete "$PROFILE\.wstrader\ws_trade_decisions.json"
    Delete "$PROFILE\.wstrader\ws_dashboard_cache.json"
    Delete "$PROFILE\.wstrader\ws_live_candles.json"
    Delete "$PROFILE\.wstrader\ws_live_trades_iq.json"
    Delete "$PROFILE\.wstrader\ws_live_trades_bullex.json"
    Delete "$PROFILE\.wstrader\ws_live_trades_casatrader.json"
    Delete "$PROFILE\.wstrader\ws_last_entry.json"
    Delete "$PROFILE\.wstrader\ws_dt_level_memory.json"
    Delete "$PROFILE\.wstrader\ws_bot.lock"
    Delete "$PROFILE\.wstrader\ws_daily_log.json"
    Delete "$PROFILE\.wstrader\ws_brain_weights.json"
    Delete "$PROFILE\.wstrader\ws_ai_stats_hs.json"
    Delete "$PROFILE\.wstrader\ws_reversal_data_unified.json"
    Delete "$PROFILE\.wstrader\daily_lockout.json"
    Delete "$PROFILE\.wstrader\loss_analysis.json"
    Delete "$PROFILE\.wstrader\hs_bot_train_control.json"
    Delete "$PROFILE\.wstrader\hs_ia_dashboard_stats.json"
    Delete "$PROFILE\.wstrader\hs_ia_train_control.json"
    Delete "$PROFILE\.wstrader\ws_ai_stats_m1.json"

    ; Stats per-ativo e per-broker (glob ws_ai_stats_*.json)
    FindFirst $0 $1 "$PROFILE\.wstrader\ws_ai_stats_*.json"
    loop_stats:
        StrCmp $1 "" done_stats
        Delete "$PROFILE\.wstrader\$1"
        FindNext $0 $1
        Goto loop_stats
    done_stats:
    FindClose $0

    ; Modelos entry_guard per-ativo (entry_guard_*.pkl)
    FindFirst $0 $1 "$PROFILE\.wstrader\entry_guard_*.pkl"
    loop_entry:
        StrCmp $1 "" done_entry
        Delete "$PROFILE\.wstrader\$1"
        FindNext $0 $1
        Goto loop_entry
    done_entry:
    FindClose $0

    ; Modelos reversal per-broker (reversal_tf_*.pkl)
    FindFirst $0 $1 "$PROFILE\.wstrader\reversal_tf_*.pkl"
    loop_reversal:
        StrCmp $1 "" done_reversal
        Delete "$PROFILE\.wstrader\$1"
        FindNext $0 $1
        Goto loop_reversal
    done_reversal:
    FindClose $0

    ; --- %APPDATA%/WsTrader/trade_memory/ ---
    RMDir /r "$APPDATA\WsTrader\trade_memory"

    ; --- Logs no %USERPROFILE% ---
    Delete "$PROFILE\wstrader_error.txt"
    Delete "$PROFILE\wstrader_backend.log"
    Delete "$PROFILE\wstrader_backend_crash.log"

    ; --- Temp update files ---
    RMDir /r "$TEMP\wstrader_update"

    ; --- Operations data in install dir ---
    RMDir /r "$INSTDIR\data"
    RMDir /r "$INSTDIR\exports"

    ; --- Legacy daily_data ---
    RMDir /r "$PROFILE\.wstrader\daily_data"

    ; ================================
    ; FIM DA LIMPEZA - .env e preferences.json preservados
    ; ================================
    
    ; Força sobrescrever SEMPRE (ignora datas/versões)
    SetOverwrite on
    SetOverwrite ifnewer
    SetOverwrite try

    ; Copia o executável OneFile
    File "dist\WsTrader.exe"

    ; Cria atalhos
    CreateDirectory "$SMPROGRAMS\WS Trader AI"
    CreateShortcut "$SMPROGRAMS\WS Trader AI\WS Trader AI.lnk" "$INSTDIR\WsTrader.exe" "" "$INSTDIR\WsTrader.exe" 0
    CreateShortcut "$SMPROGRAMS\WS Trader AI\Desinstalar.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0

    ; Atalho na área de trabalho
    CreateShortcut "$DESKTOP\WS Trader AI.lnk" "$INSTDIR\WsTrader.exe" "" "$INSTDIR\WsTrader.exe" 0

    ; Salva informações no registro
    WriteRegStr HKLM "Software\WsTrader" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "Software\WsTrader" "Version" "6.9.0"

    ; Cria desinstalador
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Adiciona no Adicionar/Remover Programas
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "DisplayName" "WS Trader AI"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "DisplayIcon" "$INSTDIR\WsTrader.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "Publisher" "WS Trader Team"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "DisplayVersion" "6.9.0"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "NoRepair" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WsTrader" "EstimatedSize" 320000

SectionEnd

; ================================
; Seção de Desinstalação
; ================================
Section "Uninstall"
    ; Remove arquivos da instalação
    RMDir /r "$INSTDIR"

    ; PRESERVA dados do usuário (~/.wstrader) — contém credenciais e memória da IA
    ; O usuário pode remover manualmente se quiser: %USERPROFILE%\.wstrader

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
