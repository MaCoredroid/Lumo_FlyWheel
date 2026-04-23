from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base


def make_session_factory(url):
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)
