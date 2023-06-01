@echo off

if "%ComputerName%"=="LAPTOP-BRAM" (
	echo Don't want to run this on the host now do we...
	@pause
	exit /B
)

echo Installing GSP...
START /WAIT GSPAPI.exe /VERYSILENT /COMPONENTS="gspmain,gspapi,bde" /TASKS=

echo Setting registry keys...
REG ADD HKCU\Software\NLR\GSP\11 /v RegGivenCode /t REG_SZ /d 4DF20CF438B305256804B73A0EFD244967BF
REG ADD HKCU\Software\NLR\GSP\11 /v RegName /t REG_SZ /d Ramdin@TUD

echo Installing Python...
START /WAIT PythonInstall.exe /quiet PrependPath=1

@pause