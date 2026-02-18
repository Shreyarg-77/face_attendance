import sqlite3

conn = sqlite3.connect('models/attendance.db')
c = conn.cursor()

# Add encoding column if missing
try:
    c.execute("ALTER TABLE students ADD COLUMN encoding TEXT;")
    print("Encoding column added.")
except sqlite3.OperationalError:
    print("Encoding column already exists.")

# Add class_display_id column if missing
try:
    c.execute("ALTER TABLE students ADD COLUMN class_display_id INTEGER;")
    print("class_display_id column added.")
except sqlite3.OperationalError:
    print("class_display_id column already exists.")

conn.commit()
conn.close()
print("Database updated successfully.")