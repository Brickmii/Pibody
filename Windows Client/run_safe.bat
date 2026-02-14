@echo off
set HOST=%1
if "%HOST%"=="" set HOST=192.168.5.24
python pbai_client.py --host %HOST% --simulate --stream-fps 2
