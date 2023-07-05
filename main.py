import requests
import asyncio
from .balance import LoadBalancer
from random import choice, randint
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dataclasses import dataclass, asdict

@dataclass
class Individual:
    prompt: str
    image: str
    pid: int
    ppid: int
    mutations: tuple = ()


llama_urls = ["http://localhost:8001/", "http://localhost:8010/"]
sd_urls = [f"http://localhost:800{i}/" for i in range(2,6)]

llambalancer = LoadBalancer(llama_urls)
sdbalancer = LoadBalancer(sd_urls)


class CrossoverPromptIdent(BaseModel):
    p1: str
    p2: str
    id: int
class PromptIdent(BaseModel):
    prompt: str
    id: int
class Ident(BaseModel):
    id: int

with open("/home/garbus/interactivediffusion/blind_promptmaker/prompts.txt", "r") as f:
    sd_prompt_list = f.readlines()

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
in_progress = defaultdict(set)
poplock = asyncio.Lock()
# keeps track of which child we've already returned
child_idx = defaultdict(int)

#async def send_to_llama(prompt):
#    full_prompt = apply_random_crossover(prompt)
#    data = {"prompt": full_prompt,
#        "use_beam_search": False,
#        "n": 1,
#        "max_tokens": 64,
#        "temperature": 0.8,
#    }
#    return await llambalancer.distribute_request(data, "LLAMA")

def apply_random_crossover(prompt1, prompt2):
    return f"""Human: Caption 1: {prompt1}
Caption 2: {prompt2}
Assistant:"""

def extract_descriptor():
    return f"""Human: Caption {choice(sd_prompt_list).strip()}
    Assistant:"""
def apply_descriptor(prompt, descriptor):
    return f"""Human: Caption: {prompt}
    Descriptor: {descriptor.strip()}
    Assistant:"""

async def make_new_prompt(prompt, prompt2=None):
    """Mutates if prompt2 not provided, else crosses over"""
    if prompt2 == None:
        extract_prompt = extract_descriptor()
        data = {"prompt": extract_prompt}
        descriptor = await llambalancer.distribute_request(data, "LLAMA", "extract")
        combine_prompt = apply_descriptor(prompt, descriptor)
        data = {"prompt": combine_prompt}
        result = await llambalancer.distribute_request(data, "LLAMA", "combine")
        return result
    data = {"prompt": apply_random_crossover(prompt, prompt2)}
    result = await llambalancer.distribute_request(data, "LLAMA", "crossover")
    return result


async def send_to_sd(prompt):
    return await sdbalancer.distribute_request({"prompt": prompt}, "SD", "")


async def add_member(ident: int, gen: int, prompt: str, prompt2:str=None):
    global popgen
    # try 3 times to generate a unique prompt
    # otherwise, just include the duplicate
    prompt = prompt.strip('"').strip()
    if prompt2:
        prompt2 = prompt2.strip('"').strip()
    new_prompt = None
    for i in range(1):
        new_prompt = await make_new_prompt(prompt, prompt2)
        new_prompt = new_prompt.strip('"').strip()
        if new_prompt != prompt:
            break
        if i < 1:
            print("============Generated the same prompt")
        else:
            print("============GIVING UP, KEEPING DUPE=======")
    new_image = await send_to_sd(new_prompt)
    pid = hash(new_prompt)
    ppid = hash(prompt.strip('"'))
    new_member = Individual(new_prompt, new_image, pid, ppid, ["crossover"])
    
    async with poplock:
        popgen[ident][gen].append(new_member)

@app.post("/genesis")
async def genesis(p: PromptIdent):
    global popgen, child_idx
    child_idx[p.id] = 0
    genesis_ind = Individual(p.prompt, "", hash(p.prompt.strip('"')),0, ["genesis"])
    popgen[p.id] = [[],[]]
    popgen[p.id][0].append(genesis_ind)

    popgen[p.id].append([])
    tasks = []
    for _ in range(16):
        tasks.append(add_member(p.id, 1, p.prompt))
    print(p.id, "submitted all tasks")
    await asyncio.gather(*tasks)
    return {"message": "Genesis Prompt submitted successfully"}

@app.post("/crossover_prompts")
async def crossover_prompts(p: CrossoverPromptIdent):
    global popgen
    append_gen = len(popgen[p.id]) - 1
    tasks = []
    for _ in range(4):
        tasks.append(add_member(p.id, append_gen, p.p1, p.p2)) # this is async
    await asyncio.gather(*tasks)
    return {"message": "Prompt submitted successfully"} 

@app.post("/submit_prompt")
async def submit_prompt(p: PromptIdent):
    global popgen
    append_gen = len(popgen[p.id]) - 1
    tasks = []
    for _ in range(4):
        tasks.append(add_member(p.id, append_gen, p.prompt)) # this is async
    await asyncio.gather(*tasks)
    return {"message": "Prompt submitted successfully"}

@app.post("/increment_generation")
async def increment_generation(ident: Ident):
    global popgen, child_idx
    # not enough parents
    if len(popgen[ident.id][0]) == 0 or (len(popgen[ident.id]) > 2 and len(popgen[ident.id][-2]) < n_parents):
        raise HTTPException(status_code=400, detail="No prompts available")
    popgen[ident.id].append([])
    child_idx[ident.id] = 0
    return {"message": "Generation incremented successfully"}


@app.get("/get_new_children")
async def get_new_children(ident: int):

    async with poplock:
        global popgen, child_idx
        print(ident, child_idx[ident], [len(pg) for pg in popgen[ident]])

        if len(popgen[ident][-2]) <= child_idx[ident]:
            raise HTTPException(status_code=204, detail="No new children available")

        new_children = popgen[ident][-2][child_idx[ident]:]
        # convert from dataclass to dictionary
        new_children = [asdict(child) for child in new_children]
        child_idx[ident] = len(popgen[ident][-2])
    return {"children": new_children}
