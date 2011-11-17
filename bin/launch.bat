@echo off
rem Add location of executing batch file to path for duration of command window.
SET BATLOC=%~dp0
PATH %BATLOC%;%PATH%
cmd /k bumps --help
