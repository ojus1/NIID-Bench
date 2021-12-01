import os
import multiprocessing as mp
import random

def run_command(args):
    i, command = args
    os.system(f"sleep {i % 10 + 2}; {command}")

scaffold = [
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)} ",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)} ",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)} ",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)} ",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)} ",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=scaffold --n_parties 5 --suffix {random.randint(0, 1000000)}",
]

# with mp.Pool(2) as p:
#     p.map(run_command, enumerate(scaffold))

fednova = [
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fednova --n_parties 5 --suffix {random.randint(0, 1000000)}",
]
# with mp.Pool(2) as p:
#     p.map(run_command, enumerate(fednova))

fedprox = [
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
]

fedprox = [
    f"python experiments.py --model=resnet --dataset=domainnet --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=resnet --dataset=domainnet --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=resnet --dataset=domainnet --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=resnet --dataset=domainnet --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=resnet --dataset=domainnet --alg=fedprox --n_parties 5 --suffix {random.randint(0, 1000000)}",
]
with mp.Pool(1) as p:
    p.map(run_command, enumerate(fedprox))

fedavg = [
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=non_iid_50_v1 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
    f"python experiments.py --model=LeNet --dataset=split_cifar10 --alg=fedavg --n_parties 5 --suffix {random.randint(0, 1000000)}",
]
# with mp.Pool(2) as p:
#     p.map(run_command, enumerate(fedavg))
