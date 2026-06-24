@echo off
cd /d C:\Users\Zac\fyp_master_starter
call .venv\Scripts\activate
set "PATH=C:\Program Files\nodejs;%PATH%"
python scripts\download_validate_dukascopy_xauusd_m1_daily.py --start 2017-05-13 --end 2026-03-31 --skip-existing
pause