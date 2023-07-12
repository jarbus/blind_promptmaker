import asyncio
from pydantic import BaseModel
from collections import defaultdict
from dataclasses import dataclass, asdict

@dataclass 
class MutationInfo:
    mutation_type: str = None
    extracted: str = None
    extracted_prompt: str = None
    crossover_prompt: str = None
    crossover_image = None
    genesis: bool = False


@dataclass
class Individual:
    prompt: str
    image: str
    pid: int
    ppid: int
    ppid2: int
    minfo: MutationInfo

# Global list to store each generation
# first element is the parent, second is the child
class Generations:
    def __init__(self):
        self.g = defaultdict(lambda: [[],[]])
        self.lock = asyncio.Lock()
    def __getitem__(self, ident):
        return self.g[ident]
    async def set_genesis(self, ident: int, genesis_ind: Individual):
        async with self.lock:
            self.g[ident] = [[],[], []]
            self.g[ident][0].append(genesis_ind)
    def get_prompt(self, ident, prompt):
        pid = hash(prompt.strip("\"").strip())
        return self.get_pid(ident, prompt)
    def get_pid(self, ident, pid):
        for i in range(len(self.g[ident])):
            for ind in self.g[ident][i]:
                if ind.pid == pid:
                    return self.g[ident][i]
        return None
    def get_lineage(self, ident, prompt):
        lineage_inds = []
        # get descendant
        descendant = gens.get_prompt(ident, prompt)
        # iterate up the tree by finding the parents
        # this is a really slow algorithm but it keeps 
        # the data model very simple
        while descendant != None and descendant.ppid != 0:
            print(descendant.ppid)
            lineage_inds.append(descendant)
            descendant = gens.get_pid(ident, descendant.ppid)

        # append genesis
        lineage_inds.append(descendant)
        return lineage_inds


    async def add_member(self, ident: int, gen: int, prompt: str, prompt2:str=None):
        """Generates and adds a new member to self.g"""
        # get rid of all quotes and trailing white space
        prompt = prompt.strip('"').strip()
        if prompt2:
            prompt2 = prompt2.strip('"').strip()
        # generate a new prompt and all mutation info
        new_prompt, minfo = await make_new_prompt(prompt, prompt2)
        new_image = await send_to_sd(new_prompt)
        pid = hash(new_prompt)
        ppid = hash(prompt.strip('"').strip())
        ppid2 = hash(prompt2.strip('"').strip()) if prompt2 else None
            
        new_member = Individual(new_prompt, new_image, pid, ppid, ppid2, minfo)
        
        if gen > 1:
            ppids = [ind.pid for ind in self.g[ident][gen-1]]
            assert ppid in ppids
        async with self.lock:
            self.g[ident][gen].append(new_member)
 

class CrossoverPromptIdent(BaseModel):
    p1: str
    p2: str
    id: int
class PromptIdent(BaseModel):
    prompt: str
    id: int
class Ident(BaseModel):
    id: int
