import sys
import requests
import asyncio
from .indmanager import IndManager, Individual, make_individual
from .mutator import Mutator
from .utils import clean
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dataclasses import dataclass, asdict


n_prompts = 16
n_parents = 4
im = IndManager("sqlite:///your_db_uri.db")
llama_urls = ["http://localhost:8001/", "http://localhost:8010/"]
sd_urls = [f"http://localhost:800{i}/" for i in range(2,10)]

mutator = Mutator(llama_urls, sd_urls)
in_progress = defaultdict(list)
prog_lock = asyncio.Lock()


async def add_member(genesis_id: int, gen: int, prompt: str, prompt2:str=None):
    """Generates and adds a new member to self.g"""
    # get rid of all quotes and trailing white space
    prompt = clean(prompt)
    prompt2 = clean(prompt2) if prompt2 else None
    ppid = hash(clean(prompt))
    ppid2 = hash(clean(prompt2)) if prompt2 else None
    async with prog_lock:
        in_progress[genesis_id].append(ppid)
    # generate a new prompt and all mutation info
    new_prompt, minfo = await mutator.make_new_prompt(prompt, prompt2)
    new_image = await mutator.send_to_sd(new_prompt)
    pid = hash(new_prompt)
    new_member = make_individual(genesis_id, new_prompt, ppid, ppid2, new_image,  gen, {"genesis": True})
    async with prog_lock:
        if ppid in in_progress[genesis_id]:
            im.add_individual(new_member)
            in_progress[genesis_id].remove(ppid)
 

class PromptGenesisID(BaseModel):
    prompt: str
    prompt2: str = None
    gen: int
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
templates = Jinja2Templates(directory="/home/garbus/interactivediffusion/blind_promptmaker/templates")
 
@app.post("/genesis")
async def genesis(p: PromptGenesisID):
    genesis_ind = make_individual(p.genesis_id, p.prompt, 0, None, "",0, {"genesis": True})
    im.add_individual(genesis_ind)

    tasks = []
    for _ in range(16):
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
async def get_new_children(genesis_id: int, gen: int, seen_pids: list = Query(None)):
    # no seen pids represented as ''
    if seen_pids:
        seen_pids = [int(pid) for pid in seen_pids[0].split(',') if pid]
    new_children = im.get_individuals_by_gen(genesis_id, gen, seen_pids)
    if len(new_children) == 0:
        raise HTTPException(status_code=204, detail="No new children available")

    # convert from dataclass to dictionary
    new_children = [child.to_dict() for child in new_children]
    for child in new_children:
        assert not child["pid"] in seen_pids

    return new_children


@app.get("/lineage", response_class=HTMLResponse)
async def get_lineage(genesis_id: int, pid: int, r: Request):
    lineage = im.get_lineage(genesis_id, pid)
    lineage = [l.to_dict() for l in lineage]
    print([l["ppid"] for l in lineage])
    # convert from bytes to string and get rid of quotes
    for l in lineage[1:]:
        l["image"] = l["image"].decode("utf-8")[1:-1]

    crossovers = []
    for ind in lineage:
        ppid2 = ind["ppid2"]
        if ppid2 != "None":
            parent2 = im.get_individuals_by_pid(genesis_id, ppid2)[0]
            parent2 = parent2.to_dict()
            parent2["image"] = parent2["image"].decode("utf-8")[1:-1]
            crossovers.append(parent2)
        else:
            crossovers.append(None)

    return templates.TemplateResponse("lineage.html", {
        "request": r,
        "image_list": lineage[1:], 
        "genesis_id": genesis_id,
        "crossover_list": crossovers,
        "genesis_prompt": lineage[0]["prompt"]})

@app.get("/working")
async def working(genesis_id: int):
    async with prog_lock:
        return len(in_progress[genesis_id])
#@app.get("/download")
#async def download(ident: int):
#    """Downloads the entire family tree as json"""
#    global gens
#    data = {'generations' : [[asdict(ind) for ind in gen]for gen in gens[ident]]}
#    return data
