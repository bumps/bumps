@echo off
rem start "Bumps Web GUI" 
call "%~dp0env\Scripts\activate.bat"
start "Bumps Webview" "python.exe" "-m" "bumps.webview.server"