import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- PASTE YOUR NEON CONNECTION STRING HERE ---
DATABASE_URL = "ABC"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to manage database sessions cleanly per request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
