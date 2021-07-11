world_rank = 9
opt_rank = 3
size = 48

n_workers = 48
n_optimizers = 4

worker_base, worker_extra = divmod(n_workers, size)
worker_split = [worker_base + (r < worker_extra) for r in range(size)]
worker_collection_ids = list(range(n_workers))[
                        sum(worker_split[:world_rank]):sum(worker_split[:world_rank + 1])]

# determine the split of worker outputs over optimizers
optimizer_base, optimizer_extra = divmod(n_workers, n_optimizers)
optimizer_split = [optimizer_base + (r < optimizer_extra) for r in range(n_optimizers)]
optimizer_collection_ids = list(range(n_workers))[
                           sum(optimizer_split[:opt_rank]):sum(optimizer_split[:opt_rank + 1])]

print(optimizer_base)
print(optimizer_extra)
print(optimizer_split)
print(f"{world_rank} -- {optimizer_collection_ids}")