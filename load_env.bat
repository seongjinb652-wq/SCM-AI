@echo off
for /f "tokens=1,2 delims==" %%a in (C:\Users\박성진\SCM-AI\.env) do (
    set %%a=%%b
)
