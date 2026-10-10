@echo off
cd /d E:\ai_video_transcript
set PORT=9000
REM Clash proxy (leave empty for direct connection; set to http://127.0.0.1:7897 for YouTube)
set YT_DLP_PROXY=
REM Whisper model size: tiny/base/small/medium/large (use medium+ for better Chinese)
set WHISPER_MODEL_SIZE=base
call venv\Scripts\activate.bat
echo ========================================
echo  AI Video Transcriber
echo  URL: http://localhost:9000
echo  Proxy: %YT_DLP_PROXY%
echo  Whisper: %WHISPER_MODEL_SIZE%
echo  Press Ctrl+C to stop
echo ========================================
python start.py --prod
pause
