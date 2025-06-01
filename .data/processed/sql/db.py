from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "postgresql://josemiguel:miguel02@localhost:5432/OlivIADbContext"

engine = create_engine(DATABASE_URL, echo=True)  # echo=True para ver SQL generado
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    Base.metadata.create_all(bind=engine)