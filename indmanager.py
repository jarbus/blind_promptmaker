import sys
import json
from .utils import clean
from random import randint
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Individual(Base):
    __tablename__ = 'individuals'
    uid = Column(Integer, primary_key=True)
    pid = Column(Integer)
    genesis_id = Column(Integer)
    prompt = Column(String)
    ppid = Column(Integer)
    ppid2 = Column(Integer)
    image = Column(String)
    gen = Column(Integer)
    mutation_info = Column(String)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

def make_individual(genesis_id: int, 
                    prompt: str, 
                    ppid: int, 
                    ppid2: int, 
                    image: str, 
                    gen: int, 
                    mutation_info: dict):
    uid = randint(0, sys.maxsize)
    pid = hash(clean(prompt))
    return Individual(uid=uid,
        pid=pid,
        genesis_id=genesis_id,
        prompt=prompt,
        ppid=ppid,
        ppid2=ppid2,
        image=image,
        gen=gen,
        mutation_info=json.dumps(mutation_info))



class IndManager:
    # TODO: figure out if it's ok to make multiple sessions like this
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_individual(self, individual):
        session = self.Session()
        session.add(individual)
        session.commit()

    def get_individuals_by_pid(self, genesis_id, pid):
        session = self.Session()
        inds = session.query(Individual).filter(Individual.genesis_id == genesis_id, Individual.pid == pid).all()
        assert len(inds) < 3 # no more than three dupes
        return inds[0]

    def get_individuals_by_gen(self, genesis_id, gen, pids):
        session = self.Session()
        return session.query(Individual).filter(Individual.genesis_id == genesis_id, Individual.gen == gen, Individual.pid.notin_(pids)).all()

    def compute_lineage(self, genesis_id, pid):
        lineage = []
        individual = self.get_individual_by_pid(genesis_id, pid)
        while individual is not None:
            lineage.append(individual)
            individual = self.get_individual_by_pid(genesis_id, individual.ppid)
        return lineage
