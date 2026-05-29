@echo off
echo ============================================================================
echo   FINDING FREE PORT AND STARTING WEB SERVER
echo ============================================================================
echo.

:: Try to find a free port starting from 9090
for %%p in (9090 9091 9092 9093 9094 9095 9096 9097 9098 9099) do (
    netstat -ano | findstr ":%%p " | findstr "LISTENING" >nul
    if errorlevel 1 (
        echo Port %%p is FREE - Starting web server...
        echo.
        echo ================================================================
        echo   WEB SERVER STARTED ON PORT %%p
        echo   Open: http://localhost:%%p
        echo ================================================================
        echo.
        python -m http.server %%p
        goto :end
    )
)

echo Could not find free port in range 9090-9099
echo Please close other applications or try different ports

:end
pause
