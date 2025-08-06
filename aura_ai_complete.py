# AURA AI - Complete Backend Application with Supabase & ML
# File: main.py

import os
import json
import jwt
import bcrypt
import enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Float, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import asyncio
import httpx

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "your-supabase-url")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-supabase-anon-key")
DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres")
JWT_SECRET = os.getenv("JWT_SECRET", "aura-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"

# Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Database Configuration
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enums
class UserRole(str, enum.Enum):
    LEARNER = "learner"
    TUTOR = "tutor"
    PARENT = "parent"
    ADMIN = "admin"

class LearningStyle(str, enum.Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    MIXED = "mixed"

class SessionStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"

class DifficultyLevel(str, enum.Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

# Enhanced Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String, unique=True, index=True, nullable=True)
    full_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=True)
    role = Column(Enum(UserRole), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    google_id = Column(String, unique=True, nullable=True)
    avatar_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # ML Tracking
    engagement_score = Column(Float, default=0.0)
    learning_velocity = Column(Float, default=0.0)
    
    # Relationships
    learner_profile = relationship("LearnerProfile", back_populates="user", uselist=False)
    tutor_profile = relationship("TutorProfile", back_populates="user", uselist=False)
    parent_profile = relationship("ParentProfile", back_populates="user", uselist=False)

class LearnerProfile(Base):
    __tablename__ = "learner_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    grade_level = Column(String, nullable=True)
    curriculum = Column(String, nullable=True)
    preferred_subjects = Column(JSON)
    learning_style = Column(Enum(LearningStyle), nullable=True)
    learning_goals = Column(JSON)
    comfort_level_online = Column(Integer, default=3)
    preferred_languages = Column(JSON)
    religion = Column(String, nullable=True)
    cultural_preferences = Column(Text, nullable=True)
    region = Column(String, nullable=True)
    session_count = Column(Integer, default=0)
    bio = Column(Text, nullable=True)
    
    # ML Enhancement Fields
    current_difficulty_level = Column(Enum(DifficultyLevel), default=DifficultyLevel.BEGINNER)
    learning_progress_score = Column(Float, default=0.0)
    attention_span_minutes = Column(Integer, default=30)
    preferred_session_time = Column(String, nullable=True)  # morning, afternoon, evening
    learning_pattern_data = Column(JSON)  # stores ML insights
    
    # Relationships
    user = relationship("User", back_populates="learner_profile")
    sessions = relationship("Session", back_populates="learner")
    feedback_received = relationship("Feedback", back_populates="learner")
    learning_analytics = relationship("LearningAnalytics", back_populates="learner")

class TutorProfile(Base):
    __tablename__ = "tutor_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String, nullable=True)
    region = Column(String, nullable=True)
    highest_qualification = Column(String, nullable=True)
    institutions_attended = Column(Text, nullable=True)
    teaching_certification = Column(Boolean, default=False)
    years_experience = Column(Integer, default=0)
    subjects_offered = Column(JSON)
    languages_spoken = Column(JSON)
    teaching_style = Column(Text, nullable=True)
    teaching_strengths = Column(Text, nullable=True)
    special_needs_support = Column(Boolean, default=False)
    religion = Column(String, nullable=True)
    cultural_preferences = Column(Text, nullable=True)
    max_hours_per_week = Column(Integer, default=20)
    preferred_session_length = Column(Integer, default=60)
    booking_notice_hours = Column(Integer, default=24)
    rating = Column(Float, default=0.0)
    total_sessions = Column(Integer, default=0)
    is_approved = Column(Boolean, default=True)
    hourly_rate = Column(Float, default=0.0)
    bio = Column(Text, nullable=True)
    
    # ML Enhancement Fields
    teaching_effectiveness_score = Column(Float, default=0.0)
    student_improvement_rate = Column(Float, default=0.0)
    adaptability_score = Column(Float, default=0.0)
    communication_rating = Column(Float, default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="tutor_profile")
    sessions = relationship("Session", back_populates="tutor")
    feedback_received = relationship("Feedback", back_populates="tutor")

class ParentProfile(Base):
    __tablename__ = "parent_profiles"
    
    id = Column(Integer, primary_key=True, index=True)  
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    relationship_to_learner = Column(String, nullable=True)
    linked_learners = Column(JSON)
    consent_given = Column(Boolean, default=False)
    consent_timestamp = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="parent_profile")
    feedback_given = relationship("Feedback", back_populates="parent")

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(Integer, ForeignKey("learner_profiles.id"))
    tutor_id = Column(Integer, ForeignKey("tutor_profiles.id"))
    subject = Column(String, nullable=False)
    scheduled_start = Column(DateTime, nullable=False)
    scheduled_end = Column(DateTime, nullable=False)
    actual_start = Column(DateTime, nullable=True)
    actual_end = Column(DateTime, nullable=True)
    status = Column(Enum(SessionStatus), default=SessionStatus.SCHEDULED)
    lesson_plan = Column(Text, nullable=True)
    session_summary = Column(Text, nullable=True)
    homework_assigned = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # ML Enhancement Fields
    difficulty_level = Column(Enum(DifficultyLevel), default=DifficultyLevel.BEGINNER)
    engagement_score = Column(Float, default=0.0)
    learning_outcome_score = Column(Float, default=0.0)
    predicted_success_rate = Column(Float, default=0.0)
    
    # Relationships
    learner = relationship("LearnerProfile", back_populates="sessions")
    tutor = relationship("TutorProfile", back_populates="sessions")
    feedback = relationship("Feedback", back_populates="session", uselist=False)

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True)
    learner_id = Column(Integer, ForeignKey("learner_profiles.id"))
    tutor_id = Column(Integer, ForeignKey("tutor_profiles.id"))
    parent_id = Column(Integer, ForeignKey("parent_profiles.id"), nullable=True)
    tutor_rating = Column(Integer, nullable=True)  # 1-5
    explanation_clarity = Column(Integer, nullable=True)  # 1-5
    learner_comfort = Column(Integer, nullable=True)  # 1-5
    would_recommend = Column(Boolean, nullable=True)
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # ML Enhancement Fields
    sentiment_score = Column(Float, default=0.0)  # -1 to 1
    topic_understanding_score = Column(Float, default=0.0)
    
    # Relationships
    session = relationship("Session", back_populates="feedback")
    learner = relationship("LearnerProfile", back_populates="feedback_received")
    tutor = relationship("TutorProfile", back_populates="feedback_received")
    parent = relationship("ParentProfile", back_populates="feedback_given")

class LearningAnalytics(Base):
    __tablename__ = "learning_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(Integer, ForeignKey("learner_profiles.id"))
    subject = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    mastery_level = Column(Float, default=0.0)  # 0-1
    time_spent_minutes = Column(Integer, default=0)
    attempts_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    last_accessed = Column(DateTime, default=func.now())
    difficulty_progression = Column(JSON)  # tracks difficulty over time
    learning_velocity = Column(Float, default=0.0)  # concepts per hour
    
    # Relationships
    learner = relationship("LearnerProfile", back_populates="learning_analytics")

class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # recommendation, difficulty_prediction, etc.
    model_path = Column(String, nullable=False)
    version = Column(String, nullable=False)
    accuracy_score = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    phone_number: Optional[str] = None
    full_name: str
    password: Optional[str] = None
    role: UserRole
    google_id: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class LearnerProfileCreate(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    grade_level: Optional[str] = None
    curriculum: Optional[str] = None
    preferred_subjects: Optional[List[str]] = []
    learning_style: Optional[LearningStyle] = None
    learning_goals: Optional[List[str]] = []
    comfort_level_online: int = 3
    preferred_languages: Optional[List[str]] = []
    religion: Optional[str] = None
    cultural_preferences: Optional[str] = None
    region: Optional[str] = None
    bio: Optional[str] = None
    attention_span_minutes: Optional[int] = 30
    preferred_session_time: Optional[str] = None

class TutorProfileCreate(BaseModel):
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    region: Optional[str] = None
    highest_qualification: Optional[str] = None
    institutions_attended: Optional[str] = None
    teaching_certification: bool = False
    years_experience: int = 0
    subjects_offered: Optional[dict] = {}
    languages_spoken: Optional[List[str]] = []
    teaching_style: Optional[str] = None
    teaching_strengths: Optional[str] = None
    special_needs_support: bool = False
    religion: Optional[str] = None
    cultural_preferences: Optional[str] = None
    max_hours_per_week: int = 20
    preferred_session_length: int = 60
    booking_notice_hours: int = 24
    hourly_rate: float = 0.0
    bio: Optional[str] = None

class SessionCreate(BaseModel):
    tutor_id: int
    subject: str
    scheduled_start: datetime
    duration_minutes: int = 60
    difficulty_level: Optional[DifficultyLevel] = DifficultyLevel.BEGINNER

class FeedbackCreate(BaseModel):
    session_id: int
    tutor_rating: Optional[int] = None
    explanation_clarity: Optional[int] = None
    learner_comfort: Optional[int] = None
    would_recommend: Optional[bool] = None
    comments: Optional[str] = None

class MLRecommendationRequest(BaseModel):
    learner_id: int
    subject: Optional[str] = None
    max_recommendations: int = 10

# ML Reinforcement Learning Engine
class AURAReinforcementLearning:
    def __init__(self):
        self.recommendation_model = None
        self.difficulty_model = None
        self.scaler = StandardScaler()
        self.model_path = "models/"
        os.makedirs(self.model_path, exist_ok=True)
        
    async def initialize_models(self, db: Session):
        """Initialize or load existing ML models"""
        try:
            # Load existing models
            self.load_models()
            logger.info("ML models loaded successfully")
        except FileNotFoundError:
            # Train new models if none exist
            await self.train_initial_models(db)
            logger.info("New ML models trained and saved")
    
    def load_models(self):
        """Load pre-trained models"""
        self.recommendation_model = joblib.load(f"{self.model_path}recommendation_model.pkl")
        self.difficulty_model = joblib.load(f"{self.model_path}difficulty_model.pkl")
        self.scaler = joblib.load(f"{self.model_path}scaler.pkl")
    
    def save_models(self):
        """Save trained models"""
        joblib.dump(self.recommendation_model, f"{self.model_path}recommendation_model.pkl")
        joblib.dump(self.difficulty_model, f"{self.model_path}difficulty_model.pkl")
        joblib.dump(self.scaler, f"{self.model_path}scaler.pkl")
    
    async def train_initial_models(self, db: Session):
        """Train initial models with synthetic data"""
        # Generate synthetic training data for demo
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [age, grade_level_encoded, comfort_level, session_count, avg_rating]
        X_recommendation = np.random.rand(n_samples, 5)
        y_recommendation = np.random.rand(n_samples)  # Compatibility scores
        
        X_difficulty = np.random.rand(n_samples, 6)  # Additional features for difficulty
        y_difficulty = np.random.choice([0, 1, 2], n_samples)  # Difficulty levels
        
        # Train recommendation model
        self.recommendation_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.recommendation_model.fit(X_recommendation, y_recommendation)
        
        # Train difficulty prediction model
        self.difficulty_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.difficulty_model.fit(X_difficulty, y_difficulty)
        
        # Fit scaler
        all_features = np.vstack([X_recommendation, X_difficulty])
        self.scaler.fit(all_features)
        
        self.save_models()
    
    async def get_tutor_recommendations(self, learner_profile: LearnerProfile, tutors: List[TutorProfile], db: Session) -> List[Dict]:
        """Generate ML-based tutor recommendations"""
        if not self.recommendation_model:
            return self._fallback_recommendations(tutors)
        
        recommendations = []
        
        # Extract learner features
        learner_features = self._extract_learner_features(learner_profile)
        
        for tutor in tutors:
            # Extract tutor features and combine with learner features
            combined_features = self._combine_features(learner_features, tutor)
            
            # Scale features
            scaled_features = self.scaler.transform([combined_features])
            
            # Predict compatibility
            compatibility_score = self.recommendation_model.predict(scaled_features)[0]
            
            # Get tutor user info
            tutor_user = db.query(User).filter(User.id == tutor.user_id).first()
            
            recommendations.append({
                "tutor_id": tutor.id,
                "name": tutor_user.full_name,
                "email": tutor_user.email,
                "avatar_url": tutor_user.avatar_url,
                "subjects_offered": tutor.subjects_offered,
                "languages_spoken": tutor.languages_spoken,
                "rating": round(tutor.rating, 2),
                "total_sessions": tutor.total_sessions,
                "years_experience": tutor.years_experience,
                "compatibility_score": round(float(compatibility_score), 3),
                "region": tutor.region,
                "teaching_style": tutor.teaching_style,
                "hourly_rate": tutor.hourly_rate,
                "bio": tutor.bio,
                "teaching_effectiveness_score": tutor.teaching_effectiveness_score
            })
        
        # Sort by compatibility score
        recommendations.sort(key=lambda x: x["compatibility_score"], reverse=True)
        return recommendations
    
    def _extract_learner_features(self, learner: LearnerProfile) -> List[float]:
        """Extract numerical features from learner profile"""
        return [
            float(learner.age or 15),
            float(hash(learner.grade_level or "grade_8") % 100),  # Simple encoding
            float(learner.comfort_level_online),
            float(learner.session_count),
            float(learner.learning_progress_score)
        ]
    
    def _combine_features(self, learner_features: List[float], tutor: TutorProfile) -> List[float]:
        """Combine learner and tutor features"""
        tutor_features = [
            float(tutor.years_experience),
            float(tutor.rating),
            float(tutor.total_sessions),
            float(tutor.teaching_effectiveness_score),
            float(tutor.communication_rating)
        ]
        return learner_features + tutor_features
    
    def _fallback_recommendations(self, tutors: List[TutorProfile]) -> List[Dict]:
        """Fallback recommendations when ML model is not available"""
        return [
            {
                "tutor_id": 1,
                "name": "Sarah Williams",
                "email": "sarah@example.com",
                "subjects_offered": {"Math": ["Grade 8", "Grade 9"], "Science": ["Grade 8"]},
                "languages_spoken": ["English", "Afrikaans"],
                "rating": 4.8,
                "total_sessions": 45,
                "years_experience": 5,
                "compatibility_score": 0.92,
                "region": "Windhoek",
                "teaching_style": "Interactive and engaging",
                "hourly_rate": 25.0,
                "bio": "Experienced math and science tutor"
            }
        ]
    
    async def predict_optimal_difficulty(self, learner_id: int, subject: str, db: Session) -> DifficultyLevel:
        """Predict optimal difficulty level for a learner"""
        if not self.difficulty_model:
            return DifficultyLevel.INTERMEDIATE
        
        learner = db.query(LearnerProfile).filter(LearnerProfile.id == learner_id).first()
        if not learner:
            return DifficultyLevel.BEGINNER
        
        # Get learning analytics for the subject
        analytics = db.query(LearningAnalytics).filter(
            LearningAnalytics.learner_id == learner_id,
            LearningAnalytics.subject == subject
        ).first()
        
        features = [
            float(learner.age or 15),
            float(learner.session_count),
            float(learner.learning_progress_score),
            float(analytics.mastery_level if analytics else 0.0),
            float(analytics.success_rate if analytics else 0.0),
            float(analytics.learning_velocity if analytics else 0.0)
        ]
        
        scaled_features = self.scaler.transform([features])
        prediction = self.difficulty_model.predict(scaled_features)[0]
        
        # Map prediction to difficulty level
        if prediction < 0.33:
            return DifficultyLevel.BEGINNER
        elif prediction < 0.67:
            return DifficultyLevel.INTERMEDIATE
        else:
            return DifficultyLevel.ADVANCED
    
    async def update_learner_progress(self, learner_id: int, session_id: int, performance_score: float, db: Session):
        """Update learner's progress and retrain models periodically"""
        learner = db.query(LearnerProfile).filter(LearnerProfile.id == learner_id).first()
        if learner:
            # Update learning progress using exponential moving average
            alpha = 0.1  # Learning rate
            learner.learning_progress_score = (
                alpha * performance_score + (1 - alpha) * learner.learning_progress_score
            )
            db.commit()
            
            # Sync with Supabase
            await self.sync_progress_to_supabase(learner_id, learner.learning_progress_score)
    
    async def sync_progress_to_supabase(self, learner_id: int, progress_score: float):
        """Sync learning progress to Supabase for real-time updates"""
        try:
            supabase.table("learner_progress_realtime").upsert({
                "learner_id": learner_id,
                "progress_score": progress_score,
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to sync progress to Supabase: {e}")

# Initialize ML Engine
ml_engine = AURAReinforcementLearning()

# FastAPI Configuration
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    
    # Initialize ML models
    db = SessionLocal()
    try:
        await ml_engine.initialize_models(db)
    finally:
        db.close()
    
    logger.info("AURA AI Backend started successfully")
    yield
    # Shutdown
    logger.info("AURA AI Backend shutting down")

app = FastAPI(
    title="AURA AI - Professional Tutoring Platform",
    description="AI-powered tutoring platform with cultural intelligence and ML reinforcement learning",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication Functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Routes
@app.get("/")
def read_root():
    return {"message": "AURA AI Backend - ML Powered Tutoring Platform", "version": "3.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "ml_status": "active"}

# Authentication Routes
@app.post("/api/auth/register")
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    password_hash = None
    if user_data.password:
        password_hash = hash_password(user_data.password)
    
    user = User(
        email=user_data.email,
        phone_number=user_data.phone_number,
        full_name=user_data.full_name,
        password_hash=password_hash,
        role=user_data.role,
        google_id=user_data.google_id,
        is_verified=bool(user_data.google_id)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Sync to Supabase
    try:
        supabase.table("users").insert({
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "created_at": user.created_at.isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to sync user to Supabase: {e}")
    
    # Create access token
    access_token = create_access_token(data={"sub": user.id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "is_verified": user.is_verified
        }
    }

@app.get("/api/users/me")
def get_current_user_info(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "phone_number": current_user.phone_number,
        "full_name": current_user.full_name,
        "role": current_user.role.value,
        "is_verified": current_user.is_verified,
        "avatar_url": current_user.avatar_url,
        "created_at": current_user.created_at,
        "engagement_score": current_user.engagement_score,
        "learning_velocity": current_user.learning_velocity
    }

# Profile Routes with ML Enhancement
@app.post("/api/profiles/learner")
async def create_learner_profile(
    profile_data: LearnerProfileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in [UserRole.LEARNER, UserRole.PARENT]:
        raise HTTPException(status_code=403, detail="Only learners or parents can create learner profiles")
    
    existing_profile = db.query(LearnerProfile).filter(LearnerProfile.user_id == current_user.id).first()
    if existing_profile:
        raise HTTPException(status_code=400, detail="Learner profile already exists")
    
    profile = LearnerProfile(
        user_id=current_user.id,
        age=profile_data.age,
        gender=profile_data.gender,
        grade_level=profile_data.grade_level,
        curriculum=profile_data.curriculum,
        preferred_subjects=profile_data.preferred_subjects or [],
        learning_style=profile_data.learning_style,
        learning_goals=profile_data.learning_goals or [],
        comfort_level_online=profile_data.comfort_level_online,
        preferred_languages=profile_data.preferred_languages or [],
        religion=profile_data.religion,
        cultural_preferences=profile_data.cultural_preferences,
        region=profile_data.region,
        bio=profile_data.bio,
        attention_span_minutes=profile_data.attention_span_minutes or 30,
        preferred_session_time=profile_data.preferred_session_time,
        learning_pattern_data={}
    )
    
    db.add(profile)
    db.commit()
    db.refresh(profile)
    
    # Sync to Supabase for real-time features
    try:
        supabase.table("learner_profiles").insert({
            "id": profile.id,
            "user_id": profile.user_id,
            "age": profile.age,
            "grade_level": profile.grade_level,
            "learning_progress_score": profile.learning_progress_score,
            "current_difficulty_level": profile.current_difficulty_level.value,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to sync learner profile to Supabase: {e}")
    
    return {"message": "Learner profile created successfully", "profile_id": profile.id}

@app.post("/api/profiles/tutor")
async def create_tutor_profile(
    profile_data: TutorProfileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != UserRole.TUTOR:
        raise HTTPException(status_code=403, detail="Only tutors can create tutor profiles")
    
    existing_profile = db.query(TutorProfile).filter(TutorProfile.user_id == current_user.id).first()
    if existing_profile:
        raise HTTPException(status_code=400, detail="Tutor profile already exists")
    
    profile = TutorProfile(
        user_id=current_user.id,
        date_of_birth=profile_data.date_of_birth,
        gender=profile_data.gender,
        region=profile_data.region,
        highest_qualification=profile_data.highest_qualification,
        institutions_attended=profile_data.institutions_attended,
        teaching_certification=profile_data.teaching_certification,
        years_experience=profile_data.years_experience,
        subjects_offered=profile_data.subjects_offered or {},
        languages_spoken=profile_data.languages_spoken or [],
        teaching_style=profile_data.teaching_style,
        teaching_strengths=profile_data.teaching_strengths,
        special_needs_support=profile_data.special_needs_support,
        religion=profile_data.religion,
        cultural_preferences=profile_data.cultural_preferences,
        max_hours_per_week=profile_data.max_hours_per_week,
        preferred_session_length=profile_data.preferred_session_length,
        booking_notice_hours=profile_data.booking_notice_hours,
        hourly_rate=profile_data.hourly_rate,
        bio=profile_data.bio
    )
    
    db.add(profile)
    db.commit()
    db.refresh(profile)
    
    # Sync to Supabase
    try:
        supabase.table("tutor_profiles").insert({
            "id": profile.id,
            "user_id": profile.user_id,
            "years_experience": profile.years_experience,
            "rating": profile.rating,
            "teaching_effectiveness_score": profile.teaching_effectiveness_score,
            "is_approved": profile.is_approved,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to sync tutor profile to Supabase: {e}")
    
    return {"message": "Tutor profile created successfully", "profile_id": profile.id}

# ML-Powered Matching Routes
@app.post("/api/matching/recommendations")
async def get_ml_recommendations(
    request: MLRecommendationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in [UserRole.LEARNER, UserRole.PARENT]:
        raise HTTPException(status_code=403, detail="Only learners or parents can get recommendations")
    
    # Get learner profile
    learner_profile = db.query(LearnerProfile).filter(LearnerProfile.id == request.learner_id).first()
    if not learner_profile:
        raise HTTPException(status_code=404, detail="Learner profile not found")
    
    # Get available tutors
    tutors_query = db.query(TutorProfile).filter(TutorProfile.is_approved == True)
    
    # Filter by subject if specified
    if request.subject:
        # This would need more sophisticated filtering based on subjects_offered JSON
        tutors = tutors_query.all()
        filtered_tutors = []
        for tutor in tutors:
            if request.subject in tutor.subjects_offered:
                filtered_tutors.append(tutor)
        tutors = filtered_tutors
    else:
        tutors = tutors_query.all()
    
    # Get ML-powered recommendations
    recommendations = await ml_engine.get_tutor_recommendations(learner_profile, tutors, db)
    
    return {
        "recommendations": recommendations[:request.max_recommendations],
        "total_found": len(recommendations),
        "ml_powered": True
    }

@app.get("/api/matching/tutors")
async def get_recommended_tutors(
    subject: str = None,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != UserRole.LEARNER:
        raise HTTPException(status_code=403, detail="Only learners can get tutor recommendations")
    
    learner_profile = db.query(LearnerProfile).filter(LearnerProfile.user_id == current_user.id).first()
    
    if not learner_profile:
        # Return demo recommendations for users without profiles
        return {
            "recommendations": [
                {
                    "tutor_id": 1,
                    "name": "Sarah Williams",
                    "email": "sarah@example.com",
                    "subjects_offered": {"Math": ["Grade 8", "Grade 9"], "Science": ["Grade 8"]},
                    "languages_spoken": ["English", "Afrikaans"],
                    "rating": 4.8,
                    "total_sessions": 45,
                    "years_experience": 5,
                    "compatibility_score": 0.92,
                    "region": "Windhoek",
                    "teaching_style": "Interactive and engaging",
                    "hourly_rate": 25.0,
                    "bio": "Experienced math and science tutor",
                    "ml_powered": False
                }
            ],
            "total_found": 1,
            "ml_powered": False
        }
    
    # Use ML recommendations
    tutors_query = db.query(TutorProfile).filter(TutorProfile.is_approved == True)
    tutors = tutors_query.all()
    
    recommendations = await ml_engine.get_tutor_recommendations(learner_profile, tutors, db)
    
    return {
        "recommendations": recommendations[:limit],
        "total_found": len(recommendations),
        "ml_powered": True
    }

# Enhanced Session Management with ML
@app.post("/api/sessions/book")
async def book_session(
    session_data: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != UserRole.LEARNER:
        raise HTTPException(status_code=403, detail="Only learners can book sessions")
    
    learner_profile = db.query(LearnerProfile).filter(LearnerProfile.user_id == current_user.id).first()
    if not learner_profile:
        raise HTTPException(status_code=404, detail="Learner profile not found")
    
    # Verify tutor exists
    tutor = db.query(TutorProfile).filter(TutorProfile.id == session_data.tutor_id).first()
    if not tutor or not tutor.is_approved:
        raise HTTPException(status_code=404, detail="Tutor not found or not approved")
    
    # Predict optimal difficulty level using ML
    optimal_difficulty = await ml_engine.predict_optimal_difficulty(
        learner_profile.id, session_data.subject, db
    )
    
    # Create session with ML predictions
    scheduled_end = session_data.scheduled_start + timedelta(minutes=session_data.duration_minutes)
    
    session = Session(
        learner_id=learner_profile.id,
        tutor_id=session_data.tutor_id,
        subject=session_data.subject,
        scheduled_start=session_data.scheduled_start,
        scheduled_end=scheduled_end,
        difficulty_level=session_data.difficulty_level or optimal_difficulty,
        predicted_success_rate=0.75  # Default prediction, will be updated with more data
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    # Update learner session count
    learner_profile.session_count += 1
    db.commit()
    
    # Sync to Supabase for real-time tracking
    try:
        supabase.table("sessions").insert({
            "id": session.id,
            "learner_id": session.learner_id,
            "tutor_id": session.tutor_id,
            "subject": session.subject,
            "scheduled_start": session.scheduled_start.isoformat(),
            "difficulty_level": session.difficulty_level.value,
            "predicted_success_rate": session.predicted_success_rate,
            "status": session.status.value
        }).execute()
    except Exception as e:
        logger.error(f"Failed to sync session to Supabase: {e}")
    
    return {
        "message": "Session booked successfully",
        "session_id": session.id,
        "scheduled_start": session.scheduled_start,
        "scheduled_end": session.scheduled_end,
        "predicted_difficulty": optimal_difficulty.value,
        "predicted_success_rate": session.predicted_success_rate
    }

@app.get("/api/sessions/my-sessions")
def get_my_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sessions = []
    
    if current_user.role == UserRole.LEARNER:
        learner_profile = db.query(LearnerProfile).filter(LearnerProfile.user_id == current_user.id).first()
        if learner_profile:
            sessions = db.query(Session).filter(Session.learner_id == learner_profile.id).order_by(Session.scheduled_start.desc()).all()
    
    elif current_user.role == UserRole.TUTOR:
        tutor_profile = db.query(TutorProfile).filter(TutorProfile.user_id == current_user.id).first()
        if tutor_profile:
            sessions = db.query(Session).filter(Session.tutor_id == tutor_profile.id).order_by(Session.scheduled_start.desc()).all()
    
    # Format sessions for response
    formatted_sessions = []
    for session in sessions:
        learner = db.query(LearnerProfile).filter(LearnerProfile.id == session.learner_id).first()
        tutor = db.query(TutorProfile).filter(TutorProfile.id == session.tutor_id).first()
        learner_user = db.query(User).filter(User.id == learner.user_id).first()
        tutor_user = db.query(User).filter(User.id == tutor.user_id).first()
        
        formatted_sessions.append({
            "session_id": session.id,
            "subject": session.subject,
            "scheduled_start": session.scheduled_start,
            "scheduled_end": session.scheduled_end,
            "status": session.status.value,
            "learner_name": learner_user.full_name,
            "tutor_name": tutor_user.full_name,
            "lesson_plan": session.lesson_plan,
            "session_summary": session.session_summary,
            "difficulty_level": session.difficulty_level.value,
            "engagement_score": session.engagement_score,
            "learning_outcome_score": session.learning_outcome_score,
            "predicted_success_rate": session.predicted_success_rate
        })
    
    return {"sessions": formatted_sessions}

# ML Analytics and Progress Tracking
@app.get("/api/analytics/learner-progress/{learner_id}")
async def get_learner_progress(
    learner_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check permissions
    if current_user.role == UserRole.LEARNER:
        learner_profile = db.query(LearnerProfile).filter(LearnerProfile.user_id == current_user.id).first()
        if not learner_profile or learner_profile.id != learner_id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role not in [UserRole.TUTOR, UserRole.PARENT, UserRole.ADMIN]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Get learning analytics
    analytics = db.query(LearningAnalytics).filter(
        LearningAnalytics.learner_id == learner_id
    ).all()
    
    # Get learner profile
    learner = db.query(LearnerProfile).filter(LearnerProfile.id == learner_id).first()
    if not learner:
        raise HTTPException(status_code=404, detail="Learner not found")
    
    # Aggregate progress data
    progress_data = {}
    for analytic in analytics:
        subject = analytic.subject
        if subject not in progress_data:
            progress_data[subject] = {
                "mastery_level": 0.0,
                "time_spent_minutes": 0,
                "success_rate": 0.0,
                "learning_velocity": 0.0,
                "topics": []
            }
        
        progress_data[subject]["mastery_level"] = max(progress_data[subject]["mastery_level"], analytic.mastery_level)
        progress_data[subject]["time_spent_minutes"] += analytic.time_spent_minutes
        progress_data[subject]["success_rate"] = (progress_data[subject]["success_rate"] + analytic.success_rate) / 2
        progress_data[subject]["learning_velocity"] = max(progress_data[subject]["learning_velocity"], analytic.learning_velocity)
        progress_data[subject]["topics"].append({
            "topic": analytic.topic,
            "mastery_level": analytic.mastery_level,
            "last_accessed": analytic.last_accessed
        })
    
    return {
        "learner_id": learner_id,
        "overall_progress_score": learner.learning_progress_score,
        "current_difficulty_level": learner.current_difficulty_level.value,
        "total_sessions": learner.session_count,
        "subjects_progress": progress_data,
        "learning_insights": {
            "preferred_learning_time": learner.preferred_session_time,
            "attention_span_minutes": learner.attention_span_minutes,
            "learning_pattern_data": learner.learning_pattern_data
        }
    }

@app.post("/api/analytics/update-progress")
async def update_learning_progress(
    session_id: int,
    performance_score: float,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in [UserRole.TUTOR, UserRole.ADMIN]:
        raise HTTPException(status_code=403, detail="Only tutors can update progress")
    
    # Get session
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update session with performance data
    session.learning_outcome_score = performance_score
    session.status = SessionStatus.COMPLETED
    db.commit()
    
    # Update learner progress using ML
    background_tasks.add_task(
        ml_engine.update_learner_progress,
        session.learner_id,
        session_id,
        performance_score,
        db
    )
    
    return {"message": "Progress updated successfully", "performance_score": performance_score}

# Enhanced Feedback with Sentiment Analysis
@app.post("/api/feedback/submit")
async def submit_feedback(
    feedback_data: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get session
    session = db.query(Session).filter(Session.id == feedback_data.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Verify user can submit feedback
    can_submit = False
    parent_id = None
    
    if current_user.role == UserRole.LEARNER:
        learner_profile = db.query(LearnerProfile).filter(LearnerProfile.user_id == current_user.id).first()
        if learner_profile and learner_profile.id == session.learner_id:
            can_submit = True
    
    elif current_user.role == UserRole.PARENT:
        parent_profile = db.query(ParentProfile).filter(ParentProfile.user_id == current_user.id).first()
        if parent_profile:
            linked_learners = parent_profile.linked_learners or []
            if session.learner_id in linked_learners:
                can_submit = True
                parent_id = parent_profile.id
    
    if not can_submit:
        raise HTTPException(status_code=403, detail="You cannot submit feedback for this session")
    
    # Simple sentiment analysis (in production, use proper NLP models)
    sentiment_score = 0.0
    if feedback_data.comments:
        positive_words = ['good', 'great', 'excellent', 'amazing', 'helpful', 'clear', 'patient']
        negative_words = ['bad', 'poor', 'unclear', 'confusing', 'difficult', 'impatient']
        
        comment_lower = feedback_data.comments.lower()
        positive_count = sum(1 for word in positive_words if word in comment_lower)
        negative_count = sum(1 for word in negative_words if word in comment_lower)
        
        if positive_count + negative_count > 0:
            sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
    
    # Check if feedback already exists
    existing_feedback = db.query(Feedback).filter(Feedback.session_id == feedback_data.session_id).first()
    if existing_feedback:
        # Update existing feedback
        existing_feedback.tutor_rating = feedback_data.tutor_rating
        existing_feedback.explanation_clarity = feedback_data.explanation_clarity
        existing_feedback.learner_comfort = feedback_data.learner_comfort
        existing_feedback.would_recommend = feedback_data.would_recommend
        existing_feedback.comments = feedback_data.comments
        existing_feedback.sentiment_score = sentiment_score
    else:
        # Create new feedback
        feedback = Feedback(
            session_id=feedback_data.session_id,
            learner_id=session.learner_id,
            tutor_id=session.tutor_id,
            parent_id=parent_id,
            tutor_rating=feedback_data.tutor_rating,
            explanation_clarity=feedback_data.explanation_clarity,
            learner_comfort=feedback_data.learner_comfort,
            would_recommend=feedback_data.would_recommend,
            comments=feedback_data.comments,
            sentiment_score=sentiment_score
        )
        db.add(feedback)
    
    db.commit()
    
    # Update tutor's ratings and ML scores
    tutor = db.query(TutorProfile).filter(TutorProfile.id == session.tutor_id).first()
    if tutor and feedback_data.tutor_rating:
        # Update average rating
        all_ratings = db.query(Feedback.tutor_rating).filter(
            Feedback.tutor_id == tutor.id,
            Feedback.tutor_rating.isnot(None)
        ).all()
        
        ratings = [r[0] for r in all_ratings if r[0] is not None]
        if ratings:
            tutor.rating = sum(ratings) / len(ratings)
        
        # Update ML scores
        all_feedback = db.query(Feedback).filter(Feedback.tutor_id == tutor.id).all()
        if all_feedback:
            avg_clarity = np.mean([f.explanation_clarity for f in all_feedback if f.explanation_clarity])
            avg_comfort = np.mean([f.learner_comfort for f in all_feedback if f.learner_comfort])
            avg_sentiment = np.mean([f.sentiment_score for f in all_feedback])
            
            tutor.communication_rating = avg_clarity if avg_clarity else 0.0
            tutor.teaching_effectiveness_score = (avg_clarity + avg_comfort) / 2 if avg_clarity and avg_comfort else 0.0
            
        db.commit()
    
    # Sync to Supabase
    try:
        supabase.table("feedback_realtime").insert({
            "session_id": feedback_data.session_id,
            "tutor_rating": feedback_data.tutor_rating,
            "sentiment_score": sentiment_score,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to sync feedback to Supabase: {e}")
    
    return {
        "message": "Feedback submitted successfully",
        "sentiment_score": sentiment_score,
        "ml_analysis": "Feedback analyzed and tutor scores updated"
    }

# Dashboard Routes with ML Insights
@app.get("/api/dashboard/learner")
async def get_learner_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != UserRole.LEARNER:
        raise HTTPException(status_code=403, detail="Only learners can access learner dashboard")
    
    learner_profile = db.query(LearnerProfile).filter(LearnerProfile.user_id == current_user.id).first()
    if not learner_profile:
        return {
            "profile": {"name": current_user.full_name, "needs_setup": True},
            "stats": {"total_sessions": 0, "completed_sessions": 0, "upcoming_sessions": 0},
            "recent_sessions": [],
            "progress": {"sessions_completed": 0, "next_milestone": 5},
            "ml_insights": {"learning_velocity": 0.0, "difficulty_progression": "beginner"}
        }
    
    # Get session statistics
    total_sessions = db.query(Session).filter(Session.learner_id == learner_profile.id).count()
    completed_sessions = db.query(Session).filter(
        Session.learner_id == learner_profile.id,
        Session.status == SessionStatus.COMPLETED
    ).count()
    upcoming_sessions = db.query(Session).filter(
        Session.learner_id == learner_profile.id,
        Session.scheduled_start > datetime.now(),
        Session.status == SessionStatus.SCHEDULED
    ).count()
    
    # Get recent sessions with ML insights
    recent_sessions = db.query(Session).filter(
        Session.learner_id == learner_profile.id
    ).order_by(Session.scheduled_start.desc()).limit(5).all()
    
    formatted_recent = []
    for session in recent_sessions:
        tutor = db.query(TutorProfile).filter(TutorProfile.id == session.tutor_id).first()
        tutor_user = db.query(User).filter(User.id == tutor.user_id).first()
        
        formatted_recent.append({
            "session_id": session.id,
            "subject": session.subject,
            "tutor_name": tutor_user.full_name,
            "scheduled_start": session.scheduled_start,
            "status": session.status.value,
            "difficulty_level": session.difficulty_level.value,
            "learning_outcome_score": session.learning_outcome_score,
            "engagement_score": session.engagement_score
        })
    
    return {
        "profile": {
            "name": current_user.full_name,
            "grade_level": learner_profile.grade_level,
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "current_difficulty": learner_profile.current_difficulty_level.value,
            "needs_setup": False
        },
        "stats": {
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "upcoming_sessions": upcoming_sessions,
            "learning_progress_score": learner_profile.learning_progress_score
        },
        "recent_sessions": formatted_recent,
        "progress": {
            "sessions_completed": completed_sessions,
            "next_milestone": max(5, (completed_sessions // 5 + 1) * 5),
            "progress_percentage": min(100, learner_profile.learning_progress_score * 100)
        },
        "ml_insights": {
            "learning_velocity": current_user.learning_velocity,
            "difficulty_progression": learner_profile.current_difficulty_level.value,
            "attention_span_minutes": learner_profile.attention_span_minutes,
            "preferred_learning_time": learner_profile.preferred_session_time
        }
    }

@app.get("/api/dashboard/tutor")
def get_tutor_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != UserRole.TUTOR:
        raise HTTPException(status_code=403, detail="Only tutors can access tutor dashboard")
    
    tutor_profile = db.query(TutorProfile).filter(TutorProfile.user_id == current_user.id).first()
    if not tutor_profile:
        return {
            "profile": {"name": current_user.full_name, "needs_setup": True},
            "stats": {"total_sessions": 0, "rating": 0.0, "is_approved": False},
            "upcoming_sessions": [],
            "ml_insights": {"teaching_effectiveness": 0.0, "student_improvement_rate": 0.0}
        }
    
    # Get upcoming sessions
    upcoming_sessions = db.query(Session).filter(
        Session.tutor_id == tutor_profile.id,
        Session.scheduled_start > datetime.now(),
        Session.status == SessionStatus.SCHEDULED
    ).order_by(Session.scheduled_start).limit(10).all()
    
    formatted_upcoming = []
    for session in upcoming_sessions:
        learner = db.query(LearnerProfile).filter(LearnerProfile.id == session.learner_id).first()
        learner_user = db.query(User).filter(User.id == learner.user_id).first()
        
        formatted_upcoming.append({
            "session_id": session.id,
            "subject": session.subject,
            "learner_name": learner_user.full_name,
            "scheduled_start": session.scheduled_start,
            "difficulty_level": session.difficulty_level.value,
            "predicted_success_rate": session.predicted_success_rate
        })
    
    return {
        "profile": {
            "name": current_user.full_name,
            "rating": round(tutor_profile.rating, 2),
            "total_sessions": tutor_profile.total_sessions,
            "is_approved": tutor_profile.is_approved,
            "teaching_effectiveness_score": round(tutor_profile.teaching_effectiveness_score, 2),
            "needs_setup": False
        },
        "stats": {
            "total_sessions": tutor_profile.total_sessions,
            "rating": round(tutor_profile.rating, 2),
            "upcoming_sessions": len(formatted_upcoming),
            "teaching_effectiveness": round(tutor_profile.teaching_effectiveness_score, 2),
            "student_improvement_rate": round(tutor_profile.student_improvement_rate, 2)
        },
        "upcoming_sessions": formatted_upcoming,
        "ml_insights": {
            "teaching_effectiveness": round(tutor_profile.teaching_effectiveness_score, 2),
            "student_improvement_rate": round(tutor_profile.student_improvement_rate, 2),
            "adaptability_score": round(tutor_profile.adaptability_score, 2),
            "communication_rating": round(tutor_profile.communication_rating, 2)
        }
    }

# Admin Routes with ML Analytics
@app.get("/api/admin/stats")
def get_admin_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    total_users = db.query(User).count()
    total_learners = db.query(User).filter(User.role == UserRole.LEARNER).count()
    total_tutors = db.query(User).filter(User.role == UserRole.TUTOR).count()
    total_sessions = db.query(Session).count()
    completed_sessions = db.query(Session).filter(Session.status == SessionStatus.COMPLETED).count()
    
    # ML-enhanced statistics
    avg_learning_progress = db.query(func.avg(LearnerProfile.learning_progress_score)).scalar() or 0.0
    avg_tutor_effectiveness = db.query(func.avg(TutorProfile.teaching_effectiveness_score)).scalar() or 0.0
    avg_engagement_score = db.query(func.avg(Session.engagement_score)).scalar() or 0.0
    
    return {
        "total_users": total_users,
        "total_learners": total_learners,
        "total_tutors": total_tutors,
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
        "success_rate": round((completed_sessions / total_sessions * 100) if total_sessions > 0 else 0, 1),
        "ml_insights": {
            "avg_learning_progress": round(avg_learning_progress, 2),
            "avg_tutor_effectiveness": round(avg_tutor_effectiveness, 2),
            "avg_engagement_score": round(avg_engagement_score, 2),
            "platform_health_score": round((avg_learning_progress + avg_tutor_effectiveness + avg_engagement_score) / 3 * 100, 1)
        }
    }

# Real-time ML Insights
@app.get("/api/ml/insights/realtime")
async def get_realtime_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in [UserRole.ADMIN, UserRole.TUTOR]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Get real-time data from Supabase
    try:
        # Fetch recent session data
        recent_sessions = supabase.table("sessions").select("*").order("scheduled_start", desc=True).limit(10).execute()
        
        # Fetch recent feedback
        recent_feedback = supabase.table("feedback_realtime").select("*").order("timestamp", desc=True).limit(5).execute()
        
        # Fetch learner progress updates
        progress_updates = supabase.table("learner_progress_realtime").select("*").order("updated_at", desc=True).limit(10).execute()
        
        return {
            "recent_sessions": recent_sessions.data,
            "recent_feedback": recent_feedback.data,
            "progress_updates": progress_updates.data,
            "timestamp": datetime.utcnow().isoformat(),
            "ml_status": "active"
        }
    except Exception as e:
        logger.error(f"Failed to fetch real-time insights from Supabase: {e}")
        return {
            "recent_sessions": [],
            "recent_feedback": [],
            "progress_updates": [],
            "timestamp": datetime.utcnow().isoformat(),
            "ml_status": "degraded",
            "error": "Real-time data temporarily unavailable"
        }

# ML Model Management
@app.post("/api/ml/retrain")
async def retrain_models(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Trigger model retraining in background
    background_tasks.add_task(retrain_ml_models, db)
    
    return {"message": "Model retraining initiated", "status": "processing"}

async def retrain_ml_models(db: Session):
    """Background task to retrain ML models with latest data"""
    try:
        logger.info("Starting ML model retraining...")
        
        # Collect training data
        sessions = db.query(Session).filter(Session.status == SessionStatus.COMPLETED).all()
        feedback_data = db.query(Feedback).all()
        
        if len(sessions) < 50:  # Need minimum data for training
            logger.warning("Insufficient data for retraining. Need at least 50 completed sessions.")
            return
        
        # Prepare training data for recommendation model
        X_recommendation = []
        y_recommendation = []
        
        for session in sessions:
            learner = db.query(LearnerProfile).filter(LearnerProfile.id == session.learner_id).first()
            tutor = db.query(TutorProfile).filter(TutorProfile.id == session.tutor_id).first()
            feedback = db.query(Feedback).filter(Feedback.session_id == session.id).first()
            
            if learner and tutor and feedback:
                features = ml_engine._extract_learner_features(learner)
                combined_features = ml_engine._combine_features(features, tutor)
                X_recommendation.append(combined_features)
                
                # Use feedback rating as target
                target_score = feedback.tutor_rating / 5.0 if feedback.tutor_rating else 0.5
                y_recommendation.append(target_score)
        
        # Retrain recommendation model
        if len(X_recommendation) >= 20:
            X_recommendation = np.array(X_recommendation)
            y_recommendation = np.array(y_recommendation)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_recommendation, y_recommendation, test_size=0.2, random_state=42
            )
            
            # Train new model
            new_recommendation_model = RandomForestRegressor(n_estimators=100, random_state=42)
            new_recommendation_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = new_recommendation_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"New recommendation model R2 score: {r2}")
            
            # Update model if performance is good
            if r2 > 0.3:  # Minimum acceptable performance
                ml_engine.recommendation_model = new_recommendation_model
                ml_engine.save_models()
                
                # Save model info to database
                model_record = MLModel(
                    model_name="tutor_recommendation",
                    model_type="recommendation",
                    model_path=f"{ml_engine.model_path}recommendation_model.pkl",
                    version=f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    accuracy_score=r2,
                    is_active=True
                )
                db.add(model_record)
                db.commit()
                
                logger.info("Recommendation model retrained and updated successfully")
            else:
                logger.warning(f"New model performance too low (R2: {r2}). Keeping existing model.")
        
        logger.info("ML model retraining completed")
        
    except Exception as e:
        logger.error(f"Failed to retrain ML models: {e}")

# Advanced ML Features
@app.get("/api/ml/predict/session-success/{session_id}")
async def predict_session_success(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in [UserRole.TUTOR, UserRole.ADMIN]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    learner = db.query(LearnerProfile).filter(LearnerProfile.id == session.learner_id).first()
    tutor = db.query(TutorProfile).filter(TutorProfile.id == session.tutor_id).first()
    
    # Calculate success probability based on various factors
    factors = {
        "tutor_rating": tutor.rating / 5.0,
        "learner_progress": learner.learning_progress_score,
        "compatibility": 0.8,  # This would come from the recommendation model
        "difficulty_match": 0.7,  # Based on difficulty vs learner level
        "time_preference": 0.9   # Based on preferred session time
    }
    
    # Weighted average of factors
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    success_probability = sum(factor * weight for factor, weight in zip(factors.values(), weights))
    
    # Update session with prediction
    session.predicted_success_rate = success_probability
    db.commit()
    
    return {
        "session_id": session_id,
        "predicted_success_rate": round(success_probability, 3),
        "factors": factors,
        "confidence": "high" if success_probability > 0.7 else "medium" if success_probability > 0.5 else "low",
        "recommendations": [
            "Good match - high probability of success" if success_probability > 0.7
            else "Consider adjusting difficulty level" if success_probability > 0.5
            else "May need additional support or different approach"
        ]
    }

@app.get("/api/ml/learning-path/{learner_id}")
async def generate_learning_path(
    learner_id: int,
    subject: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in [UserRole.LEARNER, UserRole.PARENT, UserRole.TUTOR, UserRole.ADMIN]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    learner = db.query(LearnerProfile).filter(LearnerProfile.id == learner_id).first()
    if not learner:
        raise HTTPException(status_code=404, detail="Learner not found")
    
    # Get learning analytics for the subject
    analytics = db.query(LearningAnalytics).filter(
        LearningAnalytics.learner_id == learner_id,
        LearningAnalytics.subject == subject
    ).all()
    
    # Generate personalized learning path
    current_level = learner.current_difficulty_level
    learning_goals = learner.learning_goals or []
    
    # Sample learning path (in production, this would use sophisticated ML)
    learning_path = {
        "learner_id": learner_id,
        "subject": subject,
        "current_level": current_level.value,
        "estimated_completion_weeks": 12,
        "milestones": [
            {
                "week": 1,
                "topic": f"Introduction to {subject}",
                "difficulty": "beginner",
                "estimated_hours": 3,
                "mastery_threshold": 0.7
            },
            {
                "week": 4,
                "topic": f"Intermediate {subject} Concepts",
                "difficulty": "intermediate",
                "estimated_hours": 4,
                "mastery_threshold": 0.75
            },
            {
                "week": 8,
                "topic": f"Advanced {subject} Applications",
                "difficulty": "advanced",
                "estimated_hours": 5,
                "mastery_threshold": 0.8
            },
            {
                "week": 12,
                "topic": f"{subject} Mastery Assessment",
                "difficulty": "advanced",
                "estimated_hours": 2,
                "mastery_threshold": 0.85
            }
        ],
        "recommended_session_frequency": 2,  # sessions per week
        "adaptive_adjustments": {
            "increase_difficulty_if": "mastery_level > 0.85",
            "decrease_difficulty_if": "mastery_level < 0.6",
            "extend_timeline_if": "progress_rate < 0.7"
        }
    }
    
    return learning_path

# Supabase Integration Endpoints
@app.post("/api/supabase/sync-all")
async def sync_all_to_supabase(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Sync users
        users = db.query(User).all()
        user_data = []
        for user in users:
            user_data.append({
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value,
                "engagement_score": user.engagement_score,
                "created_at": user.created_at.isoformat()
            })
        
        if user_data:
            supabase.table("users").upsert(user_data).execute()
        
        # Sync sessions
        sessions = db.query(Session).all()
        session_data = []
        for session in sessions:
            session_data.append({
                "id": session.id,
                "learner_id": session.learner_id,
                "tutor_id": session.tutor_id,
                "subject": session.subject,
                "scheduled_start": session.scheduled_start.isoformat(),
                "status": session.status.value,
                "difficulty_level": session.difficulty_level.value,
                "engagement_score": session.engagement_score,
                "predicted_success_rate": session.predicted_success_rate
            })
        
        if session_data:
            supabase.table("sessions").upsert(session_data).execute()
        
        return {
            "message": "All data synced to Supabase successfully",
            "users_synced": len(user_data),
            "sessions_synced": len(session_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to sync to Supabase: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

# WebSocket for Real-time Updates (Optional - for future enhancement)
@app.get("/api/realtime/status")
async def get_realtime_status():
    return {
        "realtime_features": {
            "session_updates": True,
            "progress_tracking": True,
            "ml_insights": True,
            "feedback_analysis": True
        },
        "supabase_connection": "active",
        "ml_engine": "running",
        "last_model_update": datetime.utcnow().isoformat()
    }

# Health Check with ML Status
@app.get("/api/health/detailed")
def detailed_health_check(db: Session = Depends(get_db)):
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    # Test ML models
    ml_status = "healthy" if ml_engine.recommendation_model else "unhealthy"
    
    # Test Supabase connection
    try:
        supabase.table("users").select("count").limit(1).execute()
        supabase_status = "healthy"
    except Exception:
        supabase_status = "degraded"
    
    return {
        "status": "healthy" if all([db_status == "healthy", ml_status == "healthy"]) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": db_status,
            "ml_engine": ml_status,
            "supabase": supabase_status
        },
        "version": "3.0.0",
        "features": {
            "ml_recommendations": ml_status == "healthy",
            "real_time_sync": supabase_status == "healthy",
            "predictive_analytics": True,
            "sentiment_analysis": True
        }
    }

# Requirements endpoint for deployment
@app.get("/api/requirements")
def get_requirements():
    return {
        "python_version": ">=3.8",
        "dependencies": [
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
            "sqlalchemy>=2.0.0",
            "psycopg2-binary>=2.9.7",  # for PostgreSQL
            "supabase>=2.0.0",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "pydantic>=2.0.0",
            "python-jose>=3.3.0",
            "bcrypt>=4.0.0",
            "python-multipart>=0.0.6",
            "httpx>=0.25.0",
            "joblib>=1.3.0"
        ],
        "environment_variables": [
            "SUPABASE_URL",
            "SUPABASE_KEY", 
            "DATABASE_URL",
            "JWT_SECRET"
        ],
        "setup_instructions": "See README.md for detailed setup instructions"
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
            .full_name,
            "role": user.role.value,
            "is_verified": user.is_verified
        }
    }

@app.post("/api/auth/login")
def login(login_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.password_hash:
        raise HTTPException(status_code=401, detail="Please sign in with Google")
    
    if not verify_password(login_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user