import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    name = Column(String(512), nullable=False)
    root_path = Column(Text, nullable=False, unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    settings_json = Column(JSON, default=dict)
    # local | drive — Drive projects use root_path like gdrive:<folder_id>
    source = Column(String(32), default="local")
    drive_folder_id = Column(String(256), nullable=True)


class ImageRecord(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, index=True)
    path = Column(Text, nullable=False, index=True)
    rel_path = Column(Text)
    drive_file_id = Column(String(256), nullable=True)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    status = Column(String(32), default="unannotated")  # unannotated, ai, verified, flagged
    species_class = Column(String(256), nullable=True)  # parent folder name = species
    annotated_at = Column(DateTime, nullable=True)
    thumbnail_key = Column(String(64), nullable=True)


class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, index=True)
    class_id = Column(Integer, default=0)
    class_name = Column(String(256), default="unknown")
    confidence = Column(Float, default=1.0)
    x = Column(Float, default=0)
    y = Column(Float, default=0)
    w = Column(Float, default=0)
    h = Column(Float, default=0)
    rotation = Column(Float, default=0)
    polygon = Column(JSON, nullable=True)
    attributes = Column(JSON, default=dict)
    source = Column(String(32), default="human")  # human, yolo, dino, sam, clip
    z_index = Column(Integer, default=0)
    locked = Column(Boolean, default=False)
    hidden = Column(Boolean, default=False)


class ClassLabel(Base):
    __tablename__ = "classes"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, index=True)
    name = Column(String(256), nullable=False)
    color = Column(String(16), default="#00F5D4")
    hotkey = Column(String(8), nullable=True)
    supercategory = Column(String(256), nullable=True)
    worms_id = Column(String(64), nullable=True)
    sort_order = Column(Integer, default=0)


class HistoryEntry(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, index=True)
    patch_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_columns()


def _migrate_columns():
    """Add columns to existing SQLite DBs without full migration framework."""
    import sqlite3

    conn = sqlite3.connect(settings.db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(images)").fetchall()}
        if "species_class" not in cols:
            conn.execute("ALTER TABLE images ADD COLUMN species_class VARCHAR(256)")
            conn.commit()
        if "drive_file_id" not in cols:
            conn.execute("ALTER TABLE images ADD COLUMN drive_file_id VARCHAR(256)")
            conn.commit()
        proj_cols = {row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
        if "source" not in proj_cols:
            conn.execute("ALTER TABLE projects ADD COLUMN source VARCHAR(32) DEFAULT 'local'")
            conn.commit()
        if "drive_folder_id" not in proj_cols:
            conn.execute("ALTER TABLE projects ADD COLUMN drive_folder_id VARCHAR(256)")
            conn.commit()
    finally:
        conn.close()


CLASS_COLORS = [
    "#00F5D4",
    "#0077B6",
    "#2ECC71",
    "#E74C3C",
    "#9B59B6",
    "#F39C12",
    "#1ABC9C",
    "#E67E22",
]


def seed_project_classes(
    db: Session,
    project_id: int,
    extra_names: list[str] | None = None,
    folder_names_only: bool = False,
    replace: bool = False,
):
    if replace:
        db.query(ClassLabel).filter(ClassLabel.project_id == project_id).delete(
            synchronize_session=False
        )
        existing: set[str] = set()
        order = 0
    else:
        existing = {c.name for c in db.query(ClassLabel).filter(ClassLabel.project_id == project_id).all()}
        order = len(existing)
    if not folder_names_only:
        for name, color, supercat in DEFAULT_MARINE_CLASSES:
            if name not in existing:
                db.add(
                    ClassLabel(
                        project_id=project_id,
                        name=name,
                        color=color,
                        supercategory=supercat,
                        sort_order=order,
                    )
                )
                order += 1
    if extra_names:
        for i, name in enumerate(extra_names):
            if name not in existing and name != "unknown":
                color = CLASS_COLORS[i % len(CLASS_COLORS)]
                db.add(
                    ClassLabel(
                        project_id=project_id,
                        name=name,
                        color=color,
                        sort_order=order,
                    )
                )
                order += 1
    db.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DEFAULT_MARINE_CLASSES = [
    ("Clownfish", "#00F5D4", "Actinopterygii"),
    ("Coral Reef", "#2ECC71", "Anthozoa"),
    ("Reef Shark", "#E74C3C", "Elasmobranchii"),
    ("Jellyfish", "#9B59B6", "Scyphozoa"),
    ("Sea Turtle", "#3498DB", "Testudines"),
    ("Ray", "#F39C12", "Elasmobranchii"),
    ("Octopus", "#E67E22", "Cephalopoda"),
    ("Moray Eel", "#1ABC9C", "Actinopterygii"),
    ("Lobster", "#C0392B", "Malacostraca"),
    ("Grouper", "#0077B6", "Actinopterygii"),
]


