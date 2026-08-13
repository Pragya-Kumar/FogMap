import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class FogDetection(Base):
    __tablename__ = "fog_detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    camera_location = Column(String, default="Van_Camera_01")
    prediction_label = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    visibility_level = Column(String, nullable=True)  
    alert_sent = Column(Boolean, default=False)       
    latency_ms = Column(Float, nullable=False)

    alerts = relationship("ActiveAlert", back_populates="detection")

class ActiveAlert(Base):
    __tablename__ = "active_alerts"

    alert_id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("fog_detections.id"), nullable=False)
    alert_level = Column(String, default="Warning")
    is_resolved = Column(Boolean, default=False)

    detection = relationship("FogDetection", back_populates="alerts")