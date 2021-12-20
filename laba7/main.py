import numpy as np

from scipy import stats

from mutils import huber, double_stage_mean


distributions = {
    "norm": stats.norm.rvs,
    "cauchy": stats.cauchy.rvs,
    "nc_mix": lambda size: 0.9 * stats.norm.rvs(size=size) + 0.1 * stats.cauchy.rvs(size=size) 
}

measures = {
    "mean": np.mean,
    "median": np.median,
    "huber": lambda x: huber(x, 1.44),
    "d_stage": double_stage_mean
}

def monte_karlo(N: int, sample_size: int, dist_grvs, measure):
    means = [measure(dist_grvs(size=sample_size)) for _ in range(N)]

    return np.mean(means), np.var(means)

def main(N : int = 10000, sample_size: int = 100):
    for dname, grvs in distributions.items():
        print(dname)
        for mname, measure in measures.items():
            mu, var = monte_karlo(N, sample_size, grvs, measure)
            print(f"\t{mname}")
            print(f"\t\tmu:\t{mu:.6f}")
            print(f"\t\tvar:\t{var:.6f}")
        print("")

if __name__ == "__main__":
    main()

