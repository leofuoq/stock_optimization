import psycopg2
from sqlalchemy import create_engine

hostname = 'localhost'
database = 'testPython'
username = 'postgres'
pwd = 'abc'
port_id = 5432

# Hàm trả về engine SQLAlchemy
def get_engine():
    engine = create_engine(f'postgresql://{username}:{pwd}@{hostname}:{port_id}/{database}')
    return engine

hostname = 'localhost'
database = 'PTDLTC'
username = 'postgres'
pwd = 'abc'
port_id = 5432

def get_connection():
    try:
        conn = psycopg2.connect(
            host=hostname,
            dbname=database,
            user=username,
            password=pwd,
            port=port_id
        )
        print("Kết nối thành công đến PostgreSQL")
        return conn
    except Exception as e:
        print("Không thể kết nối đến PostgreSQL:", e)
        return None
# Hàm trả về engine SQLAlchemy
def get_engine():
    engine = create_engine(f'postgresql://{username}:{pwd}@{hostname}:{port_id}/{database}')
    return engine