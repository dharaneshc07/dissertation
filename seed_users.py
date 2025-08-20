import psycopg2
import bcrypt

def create_user(username, password, role):
    conn = psycopg2.connect(
        dbname="receipts_db",
        user="postgres",
        password="",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
            (username, hashed, role)
        )
        conn.commit()
        print(f"✅ User '{username}' created as {role}")
    except Exception as e:
        print(f"❌ Failed to create user: {e}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    create_user("admin", "admin123", "admin")
    create_user("employee1", "pass123", "employee")
