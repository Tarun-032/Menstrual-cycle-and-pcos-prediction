@echo off
set FLASK_APP=app.py
set FLASK_ENV=development

start /B cmd /C "flask run"
timeout /t 5
start http://127.0.0.1:5000