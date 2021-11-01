@echo off
echo "Iniciando TCC-REM!"
call "%~dp0\venv_tcc\Scripts\activate.bat"
python "%~dp0\src\gui.py"
echo "Finalizado."
pause