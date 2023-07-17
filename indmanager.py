import sys
import json
from typing import Union
from .utils import clean
from random import randint
from sqlalchemy import Column, Integer, String, create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Individual(Base):
    __tablename__ = 'individuals'
    uid = Column(String, primary_key=True)
    pid = Column(String)
    genesis_id = Column(String)
    prompt = Column(String)
    ppid = Column(String)
    ppid2 = Column(String)
    image = Column(String)
    gen = Column(Integer)
    mutation_info = Column(String)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

def make_individual(genesis_id: Union[str, int], 
                    prompt: str, 
                    ppid: Union[str,int], 
                    ppid2: Union[str, int], 
                    image: str, 
                    gen: int, 
                    mutation_info: dict):
    # Javascript can't handle large numbers, so we use
    # strings to represent all IDs
    uid = randint(0, sys.maxsize)
    pid = hash(clean(prompt))
    return Individual(uid=str(uid),
        pid=str(pid),
        genesis_id=str(genesis_id),
        prompt=prompt,
        ppid=str(ppid),
        ppid2=str(ppid2),
        image=image,
        gen=gen,
        mutation_info=json.dumps(mutation_info))



class IndManager:
    # TODO: figure out if it's ok to make multiple sessions like this
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("pragma synchronous = normal;")
            cursor.execute("pragma temp_store = memory;")
            cursor.execute("pragma mmap_size = 30000000000;")
            cursor.close()



        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_individual(self, individual):
        with self.Session() as session:
            session.add(individual)
            session.commit()

    def get_individuals_by_pid(self, genesis_id, pid):
        with self.Session() as session:
            inds = session.query(Individual).filter(Individual.genesis_id == genesis_id, Individual.pid == str(pid)).all()
            assert len(inds) < 3 # no more than three dupes
            return inds

    def get_individuals_by_gen(self, genesis_id, gen, seen_pids):
        with self.Session() as session:
            return session.query(Individual).filter(Individual.genesis_id == genesis_id, Individual.gen == gen, Individual.pid.notin_(seen_pids)).all()

    def get_lineage(self, genesis_id, pid):
        lineage = []
        individual = self.get_individuals_by_pid(genesis_id, pid)

        while individual:
            lineage.append(individual[0])
            individual = self.get_individuals_by_pid(genesis_id, individual[0].ppid)
        return list(reversed(lineage))
