import requests
import asyncio
from .balance import LoadBalancer
from random import choice, randint
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
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
            print(gens.g)
    def get_prompt(self, ident, prompt):
        pid = hash(prompt.strip("\"").strip())
        return self.get_pid(ident, pid)
    def get_pid(self, ident, pid):
        for i in range(len(self.g[ident])):
            for ind in self.g[ident][i]:
                if ind.pid == pid:
                    return ind
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

    async def increment_gen(self, ident):
        async with self.lock:
            self.g[ident].append([])


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
 
n_prompts = 16
n_parents = 4
gens = Generations()
in_progress = defaultdict(set)
# keeps track of which child we've already returned
child_idx = defaultdict(int)

llama_urls = ["http://localhost:8001/", "http://localhost:8010/"]
sd_urls = [f"http://localhost:800{i}/" for i in range(2,6)]

llambalancer = LoadBalancer(llama_urls)
sdbalancer = LoadBalancer(sd_urls)

jinja_templates = Jinja2Templates(directory="templates")

with open("/home/garbus/interactivediffusion/blind_promptmaker/prompts.txt", "r") as f:
    sd_prompt_list = f.readlines()

class CrossoverPromptIdent(BaseModel):
    p1: str
    p2: str
    id: int
class PromptIdent(BaseModel):
    prompt: str
    id: int
class Ident(BaseModel):
    id: int


def apply_random_crossover(prompt1, prompt2):
    return f"""Human: Caption 1: {prompt1}
Caption 2: {prompt2}
Assistant:"""
def extract_subject(prompt):
    return f"""Human: Caption: {prompt}
Assistant:"""

def reinsert_subject(prompt, subject):
    return f"""Human: Caption: {prompt}
Subjects: {subject}
Assistant:"""

def extract_descriptor():
    # returns the caption with and without the
    # Human: Assistant: phrases
    p = choice(sd_prompt_list).strip()
    return f"""Human: Caption {p}
    Assistant:""", p
def apply_descriptor(prompt, descriptor):
    return f"""Human: Caption: {prompt}
    Descriptor: {descriptor.strip()}
    Assistant:"""

async def mutate(prompt):
    extract_prompt_with_speakers, extract_prompt = extract_descriptor()
    data = {"prompt": extract_prompt_with_speakers}
    descriptor = await llambalancer.distribute_request(data, "LLAMA", "extract")
    combine_prompt = apply_descriptor(prompt, descriptor)
    data = {"prompt": combine_prompt}
    result = await llambalancer.distribute_request(data, "LLAMA", "combine")
    minfo = MutationInfo(mutation_type="extract",extracted=descriptor, extracted_prompt=extract_prompt)
    return result, minfo

async def crossover(p1, p2):
    data = {"prompt": apply_random_crossover(p1, p2)}
    crossover_prompt = await llambalancer.distribute_request(data, "LLAMA", "crossover")
    data = {"prompt": extract_subject(p1)}
    subject = await llambalancer.distribute_request(data, "LLAMA", "subject-extract") 
    data = {"prompt": reinsert_subject(crossover_prompt, subject)}
    child = await llambalancer.distribute_request(data, "LLAMA", "subject-reinsert") 
    return child

async def make_new_prompt(prompt, prompt2=None):
    """Mutates if prompt2 not provided, else crosses over"""
    if prompt2 != None:
        new_prompt = await crossover(prompt, prompt2)
    else:
        new_prompt = prompt
    new_prompt, minfo = await mutate(new_prompt)
    new_prompt = new_prompt.strip('"').strip()
    minfo.crossover_prompt = prompt2
    return new_prompt, minfo


async def send_to_sd(prompt):
    return await sdbalancer.distribute_request({"prompt": prompt}, "SD", "")

 
origins = [
    "http://localhost",
    "http://localhost:3000", # intereactive
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Access-Control-Allow-Origin"],
)
 
@app.post("/genesis")
async def genesis(p: PromptIdent):
    global gens, child_idx
    child_idx[p.id] = 0
    genesis_ind = Individual(p.prompt, "", hash(p.prompt.strip('"')),0, 0, MutationInfo(genesis=True))
    await gens.set_genesis(p.id, genesis_ind)

    tasks = []
    for _ in range(4):
        tasks.append(gens.add_member(p.id, 1, p.prompt))
    print(p.id, "submitted all tasks")
    await asyncio.gather(*tasks)
    return {"message": "Genesis Prompt submitted successfully"}

@app.post("/crossover_prompts")
async def crossover_prompts(p: CrossoverPromptIdent):
    global gens
    append_gen = len(gens[p.id]) - 1
    tasks = []
    for _ in range(4):
        tasks.append(gens.add_member(p.id, append_gen, p.p1, p.p2)) # this is async
    await asyncio.gather(*tasks)
    return {"message": "Prompt submitted successfully"} 

@app.post("/submit_prompt")
async def submit_prompt(p: PromptIdent):
    global gens
    append_gen = len(gens[p.id]) - 1
    tasks = []
    for _ in range(4):
        tasks.append(gens.add_member(p.id, append_gen, p.prompt)) # this is async
    await asyncio.gather(*tasks)
    return {"message": "Prompt submitted successfully"}

@app.post("/increment_generation")
async def increment_generation(ident: Ident):
    global gens, child_idx
    # not enough parents
    if len(gens[ident.id][0]) == 0 or (len(gens[ident.id]) > 2 and len(gens[ident.id][-2]) < n_parents):
        raise HTTPException(status_code=400, detail="No prompts available")
    await gens.increment_gen(ident.id)
    child_idx[ident.id] = 0
    return {"message": "Generation incremented successfully"}


@app.get("/get_new_children")
async def get_new_children(ident: int):
    global gens, child_idx

    async with gens.lock:
        print(ident, child_idx[ident], [len(pg) for pg in gens[ident]])

        if len(gens[ident][-2]) <= child_idx[ident]:
            raise HTTPException(status_code=204, detail="No new children available")

        new_children = gens[ident][-2][child_idx[ident]:]
        # convert from dataclass to dictionary
        new_children = [asdict(child) for child in new_children]
        child_idx[ident] = len(gens[ident][-2])
    return {"children": new_children}

@app.get("/download")
async def download(ident: int):
    """Downloads the entire family tree as json"""
    global gens
    data = {'generations' : [[asdict(ind) for ind in gen]for gen in gens[ident]]}
    return data

@app.get("/lineage")
async def lineage(ident:int, prompt):
    lineage_inds = gens.get_lineage(ident, prompt)
    print(lineage_inds)
    if lineage_inds == [None]:
        raise HTTPException(status_code=400, detail=f"Invalid lineage")
    return {"lineage" : [asdict(ind) for ind in lineage_inds]}
