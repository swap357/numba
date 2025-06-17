REM First, setup the VS2022 environment
for /F "usebackq tokens=*" %%i in (`vswhere.exe -nologo -products * -version "[17.0,18.0)" -property installationPath`) do (
  set "VSINSTALLDIR=%%i\\"
)
if not exist "%VSINSTALLDIR%" (
  echo "Could not find VS 2022"
  exit /B 1
)

echo "Found VS 2022 in %VSINSTALLDIR%"
call "%VSINSTALLDIR%VC\Auxiliary\Build\vcvarsall.bat" x64

REM Ensure Windows system directories are in PATH for wmic and other tools
set PATH=%SystemRoot%\System32;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%

%PYTHON% setup.py build install --single-version-externally-managed --record=record.txt

exit /b %errorlevel%
