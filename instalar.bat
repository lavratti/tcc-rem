echo "Criando ambiente virtual python..."
python -m venv "%~dp0\venv_tcc"
echo "Instalando dependencias..."
"%~dp0\venv_tcc\Scripts\pip.exe" install -r "%~dp0\requirements.txt"
echo "Fim."
pause
