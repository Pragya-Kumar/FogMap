import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- PASTE YOUR NEON CONNECTION STRING HERE ---
DATABASE_URL = "postgresql://neondb_owner:npg_8nj3liRXmIeu@ep-odd-glitter-aj9b0bpb.c-3.us-east-2.aws.neon.tech/neondb?sslmode=require"

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