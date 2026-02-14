@echo off
set HOST=%1
if "%HOST%"=="" set HOST=pibody.local
python pbai_client.py --host %HOST% --stream-fps 2
