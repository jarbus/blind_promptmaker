import sys
import requests
import asyncio
from .indmanager import IndManager, Individual, make_individual
from .mutator import Mutator
from .utils import clean
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dataclasses import dataclass, asdict


n_prompts = 16
n_parents = 4
im = IndManager("sqlite:///your_db_uri.db")
llama_urls = ["http://localhost:8001/", "http://localhost:8010/"]
sd_urls = [f"http://localhost:800{i}/" for i in range(2,6)]

mutator = Mutator(llama_urls, sd_urls)

async def add_member(genesis_id: int, gen: int, prompt: str, prompt2:str=None):
    """Generates and adds a new member to self.g"""
    # get rid of all quotes and trailing white space
    prompt = clean(prompt)
    prompt2 = clean(prompt2) if prompt2 else None
    # generate a new prompt and all mutation info
    new_prompt, minfo = await mutator.make_new_prompt(prompt, prompt2)
    new_image = await mutator.send_to_sd(new_prompt)
    pid = hash(new_prompt)
    ppid = hash(clean(prompt))
    ppid2 = hash(clean(prompt2)) if prompt2 else None
    new_member = make_individual(genesis_id, new_prompt, ppid, ppid2, new_image,  gen, {"genesis": True})
    im.add_individual(new_member)
 

class PromptGenesisID(BaseModel):
    prompt: str
    prompt2: str = None
    gen: int
    genesis_id: int
class GenesisID(BaseModel):
    genesis_id: int


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
async def genesis(p: PromptGenesisID):
    genesis_ind = make_individual(p.genesis_id, p.prompt, 0, None, "",0, {"genesis": True})
    im.add_individual(genesis_ind)

    tasks = []
    for _ in range(1):
        tasks.append(add_member(p.genesis_id, 1, p.prompt))
    print(p.genesis_id, "submitted all tasks")
    await asyncio.gather(*tasks)
    return {"message": "Genesis Prompt submitted successfully"}


@app.post("/submit_prompt")
async def submit_prompt(p: PromptGenesisID):
    tasks = []
    for _ in range(4):
        tasks.append(add_member(p.genesis_id, p.gen+1, p.prompt, prompt2=p.prompt2))
    await asyncio.gather(*tasks)
    return {"message": "Prompt submitted successfully"}

@app.get("/get_new_children")
async def get_new_children(genesis_id: int, gen: int, seen_pids: list[int]=[]):
    new_children = im.get_individuals_by_gen(genesis_id, gen, seen_pids)
    if len(new_children) == 0:
        raise HTTPException(status_code=204, detail="No new children available")

    # convert from dataclass to dictionary
    new_children = [child.to_dict() for child in new_children]
    return new_children

#@app.post("/crossover_prompts")
#async def crossover_prompts(p: CrossoverPromptGenesisID):
#    global gens
#    append_gen = len(gens[p.id]) - 1
#    tasks = []
#    for _ in range(4):
#        tasks.append(gens.add_member(p.id, append_gen, p.p1, p.p2)) # this is async
#    await asyncio.gather(*tasks)
#    return {"message": "Prompt submitted successfully"} 

#@app.post("/increment_generation")
#async def increment_generation(ident: GenesisID):
#    global gens, child_idx
#    # not enough parents
#    if len(gens[ident.id][0]) == 0 or (len(gens[ident.id]) > 2 and len(gens[ident.id][-2]) < n_parents):
#        raise HTTPException(status_code=400, detail="No prompts available")
#    gens[ident.id].append([])
#    child_idx[ident.id] = 0
#    return {"message": "Generation incremented successfully"}



#@app.get("/download")
#async def download(ident: int):
#    """Downloads the entire family tree as json"""
#    global gens
#    data = {'generations' : [[asdict(ind) for ind in gen]for gen in gens[ident]]}
#    return data
