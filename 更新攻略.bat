@echo off
rem Update guide library: edit data\elden_ring\*.md first, then run this.
rem ASCII-only content on purpose: .bat files with non-ASCII get garbled by codepage.
cd /d "%~dp0"
start "" /wait "NaviRAG.exe" --rebuild
