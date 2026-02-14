@echo off
echo ============================================================
echo  PBAI Client Setup
echo ============================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo [1/3] Installing PyTorch with CUDA 12.4...
pip install torch --index-url https://download.pytorch.org/whl/cu124
if %errorlevel% neq 0 (
    echo ERROR: PyTorch install failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing remaining dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Dependency install failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Checking GPU...
python -c "import torch; gpu = torch.cuda.is_available(); name = torch.cuda.get_device_name(0) if gpu else 'NONE'; print(f'CUDA: {gpu}  GPU: {name}')"

echo.
echo Setup complete.
pause
