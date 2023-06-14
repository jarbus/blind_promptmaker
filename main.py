import requests
import asyncio
from .balance import LoadBalancer
from random import choice, randint
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

llama_urls = ["http://localhost:8001/", "http://localhost:8002", "http://localhost:8003", "http://localhost:8004"]
sd_urls = ["http://localhost:8010/"]

llambalancer = LoadBalancer(llama_urls)
sdbalancer = LoadBalancer(sd_urls)


class PromptIdent(BaseModel):
    prompt: str
    id: int
class Ident(BaseModel):
    id: int

with open("/home/garbus/genetic_prompting/genetic_prompter/prompts.txt", "r") as f:
    sd_prompt_list = f.readlines()

def apply_random_crossover(prompt):
    return f"""Human: Caption 1: {prompt}
Caption 2: {choice(sd_prompt_list).strip()}
Assistant:"""

origins = [
    "http://localhost",
    "http://localhost:3000", # intereactive
]
n_prompts = 16
n_parents = 4

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Access-Control-Allow-Origin"],
)


# Global list to store each generation
# first element is the parent, second is the child
popgen = defaultdict(lambda: [[],[]])
poplock = asyncio.Lock()
# keeps track of which child we've already returned
child_idx = defaultdict(int)

async def send_to_llama(prompt):
    full_prompt = apply_random_crossover(prompt)
    data = {"prompt": full_prompt}
    return await llambalancer.distribute_request(data, "LLAMA")

async def send_to_sd(prompt):
    return await sdbalancer.distribute_request({"prompt": prompt}, "SD")


async def add_member(id: int, gen: int, prompt: str):
    global popgen
    new_prompt = await send_to_llama(prompt)
    new_image = await send_to_sd(new_prompt)
    new_member = {"prompt": new_prompt.strip('"'), "image": new_image}
    
    async with poplock:
        popgen[id][gen].append(new_member)

@app.post("/genesis")
async def genesis(p: PromptIdent):
    global popgen, child_idx
    child_idx[p.id] = 0
    popgen[p.id][0].append(p.prompt)
    popgen[p.id].append([])
    tasks = []
    for _ in range(16):
        tasks.append(add_member(p.id, 1, p.prompt))
    print("submitted all tasks")
    await asyncio.gather(*tasks)
    print(child_idx[p.id])
    return {"message": "Genesis Prompt submitted successfully"}


@app.post("/submit_prompt")
async def submit_prompt(p: PromptIdent):
    global popgen
    popgen[p.id][-2].append(p.prompt) # add as parent
    for _ in range(4):
        r = await send_to_llama(p.prompt)

        popgen[p.id][-1].append(r) # create 4 children

    return {"message": "Prompt submitted successfully"}

@app.post("/increment_generation")
async def increment_generation(ident: Ident):
    global popgen, child_idx
    # not enought parents
    if len(popgen[ident.id][0]) == 0 or (len(popgen[ident.id]) > 2 and len(popgen[ident.id][-2]) < n_parents):
        raise HTTPException(status_code=400, detail="No prompts available")
    popgen[ident.id].append([])
    child_idx[ident.id] = 0
    return {"message": "Generation incremented successfully"}


@app.get("/get_new_children")
async def get_new_children(ident: int):
    global popgen, child_idx
    print(child_idx[ident], [len(pg) for pg in popgen[ident]])

    if len(popgen[ident][-2]) <= child_idx[ident]:
        raise HTTPException(status_code=204, detail="No new children available")

    new_children = popgen[ident][-2]#[child_idx:]
    child_idx[ident] = len(new_children)
    return {"children": new_children}
