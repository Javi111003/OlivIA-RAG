from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from db import *

class Book(Base):
    __tablename__ = 'books'
    
    Id = Column(Integer, primary_key=True)
    Tittle = Column(String(255), nullable=False)
    Author = Column(String(255), nullable=False)
    Gender = Column(String(100))
    Description = Column(Text)
    File_path = Column(String(255), nullable=False)
    Add_date = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Libro(id={self.id}, title='{self.title}', author='{self.author}')>"