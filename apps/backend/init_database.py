import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from database import create_tables, drop_tables, engine
from config import settings
from urllib.parse import urlparse
from sqlalchemy import text
import sys

def create_database():
    """Create the database if it doesn't exist."""
    db_url = settings.DATABASE_URL
    
    if '@' in db_url and db_url.count('@') > 1:
        parts = db_url.split('@')
        if len(parts) >= 3:
            protocol_user_pass = '@'.join(parts[:-1])
            host_port_db = parts[-1]
            
            if '://' in protocol_user_pass:
                protocol_part, user_pass = protocol_user_pass.split('://', 1)
                if ':' in user_pass:
                    user, password = user_pass.split(':', 1)
                    from urllib.parse import quote
                    encoded_password = quote(password, safe='')
                    db_url = f"{protocol_part}://{user}:{encoded_password}@{host_port_db}"
    
    parsed = urlparse(db_url)
    
    host = parsed.hostname
    port = parsed.port or 5432
    user = parsed.username
    password = parsed.password
    db_name = parsed.path.lstrip('/') if parsed.path else 'mindscan'
    
    print(f"Connecting to PostgreSQL at {host}:{port}")
    print(f"Database name: {db_name}")
    print(f"User: {user}")
    
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database '{db_name}' created successfully")
        else:
            print(f"Database '{db_name}' already exists")
            
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def init_tables():
    """Initialize all database tables."""
    try:
        print("Creating database tables...")
        create_tables()
        print("All tables created successfully")
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            
        print(f"📊 Created tables: {', '.join(tables)}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def reset_database():
    """Drop and recreate all tables (for development)."""
    try:
        print("Dropping all tables...")
        drop_tables()
        print("All tables dropped")
        
        print("Creating fresh tables...")
        create_tables()
        print("Fresh tables created")
        
        return True
        
    except Exception as e:
        print(f"Error resetting database: {e}")
        return False

def test_connection():
    """Test database connection and table access."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"Database connection successful")
            print(f"   PostgreSQL version: {version}")
            
            result = conn.execute(text("SELECT COUNT(*) FROM conversations"))
            count = result.fetchone()[0]
            print(f"Table access successful - {count} conversations found")
            
        return True
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def main():
    """Main initialization function."""
    print("🚀 DASS Database Initialization")
    print("=" * 50)
    
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if 'reset' in args:
        print("⚠️  RESET MODE: This will delete all existing data!")
        confirm = input("Are you sure? Type 'yes' to continue: ")
        if confirm.lower() != 'yes':
            print("Reset cancelled")
            return
        
        if not reset_database():
            sys.exit(1)
    else:
        print("1. Skipping database creation (already exists)...")
        
        print("\n2. Creating tables...")
        if not init_tables():
            sys.exit(1)
    
    print("\n3. Testing connection...")
    if not test_connection():
        sys.exit(1)
    
  
    print("Database initialization completed!")
   
    
    
    print("\nReady to start the FastAPI server:")
    print("   python main.py")

if __name__ == "__main__":
    main() 