@echo off
echo ========================================
echo Testing GPS Update Endpoint
echo ========================================
echo.
echo Sending GPS data to server...
echo.

curl -X POST http://localhost:5000/gps/update ^
  -H "Content-Type: application/json" ^
  -d "{\"latitude\":11.0286,\"longitude\":77.0269,\"speed\":25.5,\"altitude\":420.0,\"satellites\":10}"

echo.
echo.
echo ========================================
echo Checking GPS data...
echo ========================================
echo.

curl http://localhost:5000/gps

echo.
echo.
echo ========================================
echo If status shows "active" and coordinates
echo are updated, then WiFi GPS is working!
echo ========================================
pause
