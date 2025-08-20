import psycopg2

def init_db():
    conn = psycopg2.connect(
        dbname="receipts_db",
        user="postgres",
        password="",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    # Create users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        );
    """)

    # Create receipts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id SERIAL PRIMARY KEY,
            username TEXT,
            merchant TEXT,
            date TEXT,
            time TEXT,
            amount TEXT,
            category TEXT,
            was_corrected BOOLEAN,
            image_path TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Users and receipts tables created successfully.")

if __name__ == "__main__":
    init_db()
