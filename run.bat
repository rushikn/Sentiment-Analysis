@echo off
echo Starting Backend...
start cmd /k "cd backend && python server.py"
echo Opening Frontend...
start chrome http://127.0.0.1:5000/
exit
