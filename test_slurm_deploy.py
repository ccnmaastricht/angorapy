# trainer.py
import os
import sys
import time
from collections import Counter

import ray


@ray.remote
def f():
    time.sleep(1)
    return ray.services.get_node_ip_address()


redis_password = sys.argv[1]
num_cpus = int(sys.argv[2])

print(redis_password)
print(num_cpus)
print(os.environ["ip_head"])

ray.init(address=os.environ["ip_head"], redis_password=redis_password)

print("Nodes in the Ray cluster:")
print(ray.nodes())

# The following takes one second (assuming that ray was able to access all of the allocated nodes).
start_total = time.time()
for i in range(60):
    start = time.time()
    ip_addresses = ray.get([f.remote() for _ in range(num_cpus)])
    print(Counter(ip_addresses))
    end = time.time()
    print(end - start)
print(time.time() - start_total)