@echo off
cd /d C:\Users\Zac\fyp_master_starter
call .venv\Scripts\activate
set "PATH=C:\Program Files\nodejs;%PATH%"
python scripts\rerun_dukascopy_review_days.py --audit-csv logs\dukascopy_xauusd_m1_daily_audit_1.csv
pause
