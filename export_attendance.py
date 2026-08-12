import psycopg2
import pandas as pd
from datetime import datetime
import warnings

# Suppress pandas SQL warnings
warnings.filterwarnings('ignore', category=UserWarning)

PG_DBNAME = "attendance_db"
PG_USER   = "postgres"
PG_PASS   = "YOUR_DB_PASSWORD"
PG_HOST   = "localhost"

def export_attendance_to_excel():
    print("Connecting to the PostgreSQL database...")
    try:
        conn = psycopg2.connect(
            dbname=PG_DBNAME, user=PG_USER, password=PG_PASS, host=PG_HOST
        )
    except Exception as e:
        print(f"[ERROR] Failed to connect to database: {e}")
        return

    query = """
    SELECT date AS "Date",
           name AS "Name", 
           in_time AS "In Time", 
           out_time AS "Out Time",
           status AS "Status"
    FROM attendance_sessions 
    ORDER BY in_time DESC
    """
    
    print("Fetching attendance records...")
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"[ERROR] Failed to fetch data. Have you run the tracker yet? ({e})")
        conn.close()
        return
        
    conn.close()

    if df.empty:
        print("No attendance records found to export.")
        return

    # Keep raw datetimes for the duration calc, before any string formatting.
    in_dt = pd.to_datetime(df['In Time'])
    out_dt = pd.to_datetime(df['Out Time'])

    # Duration column — computed from the raw timestamps, NOT the formatted strings.
    duration_mins = (out_dt - in_dt).dt.total_seconds() / 60
    df['Duration (Minutes)'] = duration_mins.round(1).fillna("Running...")

    # Date gets its own column (DD-MM-YYYY), separate from the time-of-day columns.
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d-%m-%Y')

    # In Time / Out Time show time-of-day only — the date already lives in its own column.
    df['In Time'] = in_dt.dt.strftime('%H:%M:%S')
    df['Out Time'] = out_dt.dt.strftime('%H:%M:%S')
    df['Out Time'] = df['Out Time'].fillna('Still IN (Timer Active)')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"Attendance_Report_{timestamp}.xlsx"
    
    print(f"Exporting {len(df)} records...")
    try:
        df.to_excel(excel_filename, index=False, engine='openpyxl')
        print(f"✅ Successfully exported to {excel_filename}")
    except ModuleNotFoundError:
        print("\n[WARNING] 'openpyxl' module is not installed. Exporting as CSV instead (opens natively in Excel).")
        csv_filename = f"Attendance_Report_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"✅ Successfully exported to {csv_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")

if __name__ == "__main__":
    export_attendance_to_excel()