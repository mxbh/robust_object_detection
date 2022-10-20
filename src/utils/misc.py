import torch
import hashlib

assert torch.rand(1, generator=torch.Generator().manual_seed(0)).item() == 0.49625658988952637,\
    'Random seeding is different on this machine compared to our experiments'
    
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        raise NotImplementedError

def get_random_generator(seed):
    # pytorch: do not use numbers with too many zero bytes
    assert isinstance(seed, str) or isinstance(seed, int) or isinstance(seed, float)
    generator = torch.Generator()
    seed = str(seed)
    seed = int(hashlib.md5(seed.encode()).hexdigest()[:16], 16) # only take half the hash to avoid overflows
    generator.manual_seed(seed)

    return generator
