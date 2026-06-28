import datetime
from database import SessionLocal, engine
import models

# 1. Force table creation check
print("🔄 Checking database structure...")
models.Base.metadata.create_all(bind=engine)

# 2. Open a temporary database session
db = SessionLocal()

try:
    print("📝 Inserting a mock AI detection log...")
    # Create a dummy test entry matching your notebook schema
    test_log = models.FogDetection(
        timestamp=datetime.datetime.utcnow(),
        prediction_label="Smog",
        confidence_score=94.5,
        visibility_level="Low",
        camera_location="Test_Bench_Van",
        alert_sent=True,
        latency_ms=45
    )
    
    db.add(test_log)
    db.commit()
    db.refresh(test_log)
    print(f"✅ Success! Row inserted with ID: #{test_log.id}")

    # 3. Read it back to verify CRUD read works
    print("📖 Reading logs back from Neon Cloud...")
    records = db.query(models.FogDetection).all()
    print(f"📊 Total records found in your cloud table: {len(records)}")
    for r in records[-3:]:  # Show the last 3 entries
        print(f"   - [ID #{r.id}] {r.prediction_label} ({r.confidence_score}%) | Visibility: {r.visibility_level}")

except Exception as e:
    print(f"❌ Verification failed! Error: {e}")

finally:
    db.close()
    print("🔒 Database connection safely closed.")