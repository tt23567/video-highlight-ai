@echo off
cd /d "D:\vscode\Video Highlight Ai Mvp"
call .venv\Scripts\activate
python -m streamlit run app.py
pause