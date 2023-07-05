import os.path as osp
import asyncio
import aiohttp

class LoadBalancer:
    def __init__(self, server_list: list):
        self.server_list = server_list
        self.server_locks = [asyncio.Lock() for _ in server_list]

    async def distribute_request(self, data: dict, t: str, endpoint):
        """Submit a request to the first server with an open lock.
        If no servers are unlocked, try again after a short wait."""
        while True:
            for lock, server in zip(self.server_locks, self.server_list):
                if lock.locked():
                    continue
                async with lock:
                    if t == "SD":
                        print(f"{server} recevied prompt")
                        async with aiohttp.ClientSession() as session:
                            async with session.get(osp.join(server,endpoint), params=data) as response:
                                return await response.read()
                    elif t == "LLAMA":
                        print(f"{server} recevied prompt")
                        async with aiohttp.ClientSession() as session:
                            async with session.post(osp.join(server, endpoint), json=data) as response:
                                text = await response.text()
                                return text.strip('"').strip(" ")
            # Wait before trying to find an available server
            await asyncio.sleep(0.3)
