@ECHO OFF
setlocal
pushd %~dp0

REM Compatibility shortcuts; build_docs.py owns the documentation workflow.
if "%PYTHON%" == "" set PYTHON=poetry run python
if "%1" == "html" goto html
if "%1" == "livehtml" goto livehtml
if "%1" == "clean" goto clean
%PYTHON% build_docs.py --help
goto end

:html
%PYTHON% build_docs.py --no-autobuild
goto end

:livehtml
%PYTHON% build_docs.py
goto end

:clean
%PYTHON% build_docs.py --clean-only

:end
set BUILD_EXIT_CODE=%ERRORLEVEL%
popd
exit /b %BUILD_EXIT_CODE%
