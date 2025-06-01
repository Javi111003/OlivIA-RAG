from sqlalchemy.orm import Session
from models import Book, SessionLocal
from typing import List, Optional

def post_book(tittle: str, 
              author: str, 
              path: str, 
              gender: Optional[str] = None, 
              description: Optional[str] = None) -> Book:
    
    db = SessionLocal()
    try:
        book = Book(
            Tittle = tittle,
            Author = author,
            File_path = path,
            Gender = gender,
            Description = description
        )
        db.add(book)
        db.commit()
        db.refresh(book)
        return book
    finally:
        db.close()

def get_book_by_id(book_id: int) -> Optional[Book]:
    db = SessionLocal()
    try:
        return db.query(Book).filter(Book.Id == book_id).first()
    finally:
        db.close()

def get_all_books(book_tittle: Optional[str] = None,
                  book_author: Optional[str] = None,
                  book_gender: Optional[str] = None,
                  limit: int = 10):
    db = SessionLocal()
    try:
        query = db.query(Book)
        if book_tittle:
            query = query.filter(Book.Tittle.ilike(f"%{book_tittle}%"))
        if book_author:
            query = query.filter(Book.Author.ilike(f"%{book_author}%"))
        if book_gender:
            query = query.filter(Book.Gender.ilike(f"%{book_gender}%"))
        return query.limit(limit).all()
    finally:
        db.close