from .balance import LoadBalancer
from random import choice
from dataclasses import dataclass, asdict

@dataclass
class MutationInfo:
    mutation_type: str
    extracted: str
    extracted_prompt: str
    crossover_prompt: str = None

class Mutator:
    def __init__(self, llama_urls, sd_urls):

        self.llambalancer = LoadBalancer(llama_urls)
        self.sdbalancer = LoadBalancer(sd_urls)

        with open("/home/garbus/interactivediffusion/blind_promptmaker/prompts.txt", "r") as f:
            self.sd_prompt_list = f.readlines()
     
    def apply_random_crossover(self, prompt1, prompt2):
        return f"""Human: Caption 1: {prompt1}
    Caption 2: {prompt2}
    Assistant:"""
    def extract_subject(self, prompt):
        return f"""Human: Caption: {prompt}
    Assistant:"""

    def reinsert_subject(self,prompt, subject):
        return f"""Human: Caption: {prompt}
    Subjects: {subject}
    Assistant:"""

    def extract_descriptor(self):
        # returns the caption with and without the
        # Human: Assistant: phrases
        p = choice(self.sd_prompt_list).strip()
        return f"""Human: Caption {p}
        Assistant:""", p
    def apply_descriptor(self, prompt, descriptor):
        return f"""Human: Caption: {prompt}
        Descriptor: {descriptor.strip()}
        Assistant:"""

    async def mutate(self, prompt):
        extract_prompt_with_speakers, extract_prompt = self.extract_descriptor()
        data = {"prompt": extract_prompt_with_speakers}
        descriptor = await self.llambalancer.distribute_request(data, "LLAMA", "extract")
        combine_prompt = self.apply_descriptor(prompt, descriptor)
        data = {"prompt": combine_prompt}
        result = await self.llambalancer.distribute_request(data, "LLAMA", "combine")
        minfo = MutationInfo(mutation_type="extract",extracted=descriptor, extracted_prompt=extract_prompt)
        return result, minfo

    async def crossover(self, p1, p2):
        data = {"prompt": self.apply_random_crossover(p1, p2)}
        crossover_prompt = await self.llambalancer.distribute_request(data, "LLAMA", "crossover")
        data = {"prompt": self.extract_subject(p1)}
        subject = await self.llambalancer.distribute_request(data, "LLAMA", "subject-extract") 
        data = {"prompt": self.reinsert_subject(crossover_prompt, subject)}
        child = await self.llambalancer.distribute_request(data, "LLAMA", "subject-reinsert") 
        return child

    async def make_new_prompt(self, prompt, prompt2=None):
        """Mutates if prompt2 not provided, else crosses over"""
        if prompt2 != None:
            new_prompt = await self.crossover(prompt, prompt2)
        else:
            new_prompt = prompt
        new_prompt, minfo = await self.mutate(new_prompt)
        new_prompt = new_prompt.strip('"').strip()
        minfo.crossover_prompt = prompt2
        return new_prompt, asdict(minfo)


    async def send_to_sd(self, prompt):
        return await self.sdbalancer.distribute_request({"prompt": prompt}, "SD", "")

     
