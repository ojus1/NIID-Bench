import torch

comm_dict = {}
def compute_comm_cost(tensor):
    memo_key = str(list(tensor.shape))
    if memo_key in comm_dict:
        return comm_dict[memo_key]

    s = tensor.numel() * 4 * 1e-6
    comm_dict[memo_key] = s
    return s

if __name__ == "__main__":
    print(compute_comm_cost(torch.randn(32, 32)))