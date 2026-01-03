import time
import math
import torch
import matplotlib.pyplot as plt

from src.layers.cacheAttention import AttentionOptimized


@torch.no_grad()
def naiveAttention(x, Wq, Wk, Wv):
    Q=x@Wq
    K=x@Wk
    V=x@Wv
    scores=Q@K.T
    attn=torch.softmax(scores/math.sqrt(Wq.shape[1]), dim=-1)
    return attn@V


def benchAppend(Tmax, D, dk, dv, repeats, warmup, threads, seed):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)

    a=AttentionOptimized(D, dk, dv)
    tokens=torch.randn(Tmax, D)

    @torch.no_grad()
    def runOpt():
        a.reset()
        a(tokens[:1])
        for t in range(2, Tmax+1):
            a(tokens[:t])

    @torch.no_grad()
    def runNaive():
        for t in range(1, Tmax+1):
            naiveAttention(tokens[:t], a.W_query, a.W_key, a.W_value)

    for _ in range(warmup):
        runOpt()
        runNaive()

    optTimes=[]
    naiveTimes=[]

    for _ in range(repeats):
        t0=time.perf_counter()
        runOpt()
        optTimes.append(time.perf_counter()-t0)

        t0=time.perf_counter()
        runNaive()
        naiveTimes.append(time.perf_counter()-t0)

    return optTimes, naiveTimes


def main(argc: int, *argv: str)->int:
    Tmax=512
    D=256
    dk=256
    dv=256
    repeats=5
    warmup=1
    threads=1
    seed=0

    optTimes, naiveTimes=benchAppend(
        Tmax, D, dk, dv, repeats, warmup, threads, seed
    )

    x=torch.arange(repeats)
    w=0.35

    plt.figure()
    plt.bar(x-w/2, optTimes, width=w, label="cached")
    plt.bar(x+w/2, naiveTimes, width=w, label="naive")
    plt.xticks(x, [f"run {i}" for i in range(repeats)])
    plt.ylabel("seconds")
    plt.title(f"T={Tmax}, D={D}")
    plt.legend()
    plt.show()

    return 0


if __name__=="__main__":
    argv=__import__("sys").argv
    exit(main(len(argv), *argv))
