import torch
import numpy as np
from zmq import device

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x+y
    z = z.to("cpu")
    # z.numpy()
    print(x, y, z)
