import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, comb
from scipy.optimize import minimize_scalar

# parameters
Miss_allignment_cases = [(0.01, 0.01), (0.05, 0.05), (0.1, 0.1)]  # (delta_ph, delta_pol)
pd = 1e-8  # dark-count probability
fec = 1.16  # error-correction inefficiency
Ncut = 40  # truncation of cat-state sums
B_Max = 20  # max AD block size
Db_Per_Km = 0.2
Loss_Step = 1.0  # dB
Loss_Max = 150.0  # dB

Known_S = [(0, 0), (0, 2), (2, 0), (2, 2), (0, 4), (4, 0), (1, 1), (1, 3), (3, 1)]
SJ = {j: [((a - j) // 2, (b - j) // 2) for a, b in Known_S if a % 2 == j and b % 2 == j]
      for j in (0, 1)}
Mu_Grid = np.concatenate([np.linspace(1e-4, 0.02, 25),
                          np.linspace(0.02, 0.06, 25),
                          np.linspace(0.06, 0.4, 20)])


# Equations

def h(x):
    if x <= 0 or x >= 1:
        return 0.0
    if x >= 0.5:
        return 1.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)  # binary entropy


# tail bound
def tail_bound(mu, j):
    n0 = 2 * (Ncut + 1) + j
    r = mu / np.sqrt((n0 + 1) * (n0 + 2))
    return c(mu, n0) / (1 - r)


def c(mu, n):
    return np.exp(-mu / 2) * mu ** (n / 2) / np.sqrt(float(factorial(n)))


def p_cond(kc, bA, bB, g, Om):
    sign = -1 if (kc ^ bA ^ bB) == 1 else +1
    q = np.exp(-g * (1 + sign * Om)) - np.exp(-2 * g)
    return (1 - pd) * (pd * np.exp(-2 * g) + q)


# Eq. (30) of CAL
def pXX(kc, g, Om):
    return 0.5 * (1 - pd) * (np.exp(-g * Om) + np.exp(g * Om)) * np.exp(-g) \
        - (1 - pd) ** 2 * np.exp(-2 * g)


#  Eq. (31) of CAL written exactly (same for (1,0) and (0,1))
def eX(kc, g, Om):
    return (np.exp(-g * Om) - (1 - pd) * np.exp(-g)) / \
        (np.exp(-g * Om) + np.exp(g * Om) - 2 * (1 - pd) * np.exp(-g))


def qZZ(nA, nB, eta, tA, tB):
    t = np.sqrt(eta)
    cA, sA, cB, sB = np.cos(tA), np.sin(tA), np.cos(tB), np.sin(tB)
    total = 0.0
    for k in range(nA + 1):
        for l in range(nB + 1):
            pref = t ** (k + l) * (1 - t) ** (nA + nB - k - l) \
                   / (2 ** (k + l) * factorial(k) * factorial(l))
            inner = 0.0
            for m in range(k + 1):
                for p in range(l + 1):
                    for r in range(max(0, m + p - l), min(k, m + p) + 1):
                        inner += (comb(k, m) * comb(l, p) * comb(k, r) * comb(l, m + p - r)
                                  * factorial(m + p) * factorial(k + l - m - p)
                                  * cA ** (m + r) * cB ** (m + 2 * p - r)
                                  * sA ** (2 * k - m - r) * sB ** (2 * l - m - 2 * p + r))
            total += comb(nA, k) * comb(nB, l) * pref * inner
    return total - (1 - t) ** (nA + nB)


def pzz_cache(eta, tA, tB):
    cache = {}
    for j in (0, 1):
        for mA, mB in SJ[j]:
            nA, nB = 2 * mA + j, 2 * mB + j
            cache[(nA, nB)] = (1 - pd) * (pd * (1 - np.sqrt(eta)) ** (nA + nB)
                                          + qZZ(nA, nB, eta, tA, tB))
    return cache


def eZ_upp(mu, pxx, cache):
    total = 0.0
    for j in (0, 1):
        pairs = [(2 * mA + j, 2 * mB + j) for mA, mB in SJ[j]]
        csum = sum(c(mu, 2 * m + j) for m in range(Ncut + 1)) + tail_bound(mu, j)
        Delta = max(csum ** 2 - sum(c(mu, a) * c(mu, b) for a, b in pairs), 0.0)
        bracket = sum(c(mu, a) * c(mu, b) * np.sqrt(max(cache[(a, b)], 0.0))
                      for a, b in pairs)
        total += (bracket + Delta) ** 2
    return total / pxx


# Advantage distillation (q = 0)
def rate_k(pxx, eb, ep, b):
    ep = min(0.5, ep)
    if b == 1:
        pAD, eb2, ep2 = 1.0, eb, ep
    else:
        pAD = (1 - eb) ** b + eb ** b
        eb2 = eb ** b / pAD
        ep2 = ((1 - eb) ** b - (1 - eb - 2 * ep) ** b) / (2 * pAD)
        ep2 = min(max(ep2, 0.0), 0.5)
    bracket = 1 - fec * h(eb2) - h(ep2)
    return (pxx / b) * pAD * bracket if bracket >= 1e-12 else 0.0


def rate(mu, eta, cache, Om, bmax=B_Max):
    g = np.sqrt(eta) * mu
    ann = []
    for kc in (1, 0):  # announcements (1,0) and (0,1)
        pxx = pXX(kc, g, Om)
        if pxx <= 0:
            return 0.0
        ann.append((pxx, eX(kc, g, Om), eZ_upp(mu, pxx, cache)))
    return max(sum(rate_k(*a, b) for a in ann) for b in range(1, bmax + 1))


def best_rate(L, d_ph, d_pol, bmax=B_Max):
    eta = 10 ** (-L / 10)
    tA = np.arcsin(np.sqrt(d_pol));
    tB = -tA
    Om = np.cos(d_ph * np.pi) * np.cos(tA - tB)
    cache = pzz_cache(eta, tA, tB)
    vals = [rate(mu, eta, cache, Om, bmax) for mu in Mu_Grid]
    i = int(np.argmax(vals))
    lo, hi = Mu_Grid[max(i - 1, 0)], Mu_Grid[min(i + 1, len(Mu_Grid) - 1)]
    res = minimize_scalar(lambda mu: -rate(mu, eta, cache, Om, bmax),
                          bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
    return max(vals[i], -res.fun)


SKR_THRESHOLD = 1e-15  # bits per round


# Loose scan on the Loss_Step grid until the rate is no longer positive.
def sweep(d_ph, d_pol, bmax):
    Ls, Rs = [], []
    for L in np.arange(0, Loss_Max + Loss_Step, Loss_Step):
        r = best_rate(L, d_ph, d_pol, bmax)
        if r <= 0:
            break
        Ls.append(L);
        Rs.append(r)
    return np.array(Ls), np.array(Rs)


# Largest loss (dB) with rate > 0 (thr = 0) or rate >= thr (thr > 0).
def find_boundary(d_ph, d_pol, L_last, bmax, thr=0.0, tol=1e-4):
    ok = (lambda r: r > 0) if thr == 0 else (lambda r: r >= thr)
    lo, hi = L_last, L_last + Loss_Step
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(best_rate(mid, d_ph, d_pol, bmax)):
            lo = mid
        else:
            hi = mid
    return lo

# Same as best_rate, but also returns mu, b, e_b and e_p^U at the optimum.
def best_point(L, d_ph, d_pol, bmax = B_Max):
    eta = 10 ** (-L / 10)
    tA = np.arcsin(np.sqrt(d_pol)); tB = -tA
    Om = np.cos(d_ph * np.pi) * np.cos(tA - tB)
    cache = pzz_cache(eta, tA, tB)
    vals = [rate(mu, eta, cache, Om, bmax) for mu in Mu_Grid]
    i = int(np.argmax(vals))
    lo, hi = Mu_Grid[max(i - 1, 0)], Mu_Grid[min(i + 1, len(Mu_Grid) - 1)]
    res = minimize_scalar(lambda mu: -rate(mu, eta, cache, Om, bmax),
                          bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
    R, mu = (-res.fun, res.x) if -res.fun > vals[i] else (vals[i], Mu_Grid[i])

    # block size and error rates at the optimum (announcement (1,0); (0,1) is identical)
    g = np.sqrt(eta) * mu
    ann = []
    for kc in (1, 0):
        pxx = pXX(kc, g, Om)
        ann.append((pxx, eX(kc, g, Om), eZ_upp(mu, pxx, cache)))
    Rb = [sum(rate_k(*a, b) for a in ann) for b in range(1, bmax + 1)]
    b = int(np.argmax(Rb)) + 1 if R > 0 else 0
    return R, mu, b, ann[0][1], min(0.5, ann[0][2])


# Sweep and plot
if __name__ == "__main__":
    plt.figure(figsize=(9, 7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rows = []

    for idx, (d_ph, d_pol) in enumerate(Miss_allignment_cases):
        col = colors[idx % len(colors)]
        for bmax, style in ((1, "--"), (B_Max, "-")):  # no AD, then AD
            Ls, Rs = sweep(d_ph, d_pol, bmax)
            cut = find_boundary(d_ph, d_pol, Ls[-1], bmax)
            L_thr = find_boundary(d_ph, d_pol, Ls[Rs >= SKR_THRESHOLD].max(), bmax,
                                  thr=SKR_THRESHOLD)  # SKR >= 1e-15
            rows.append((d_ph, d_pol, bmax, cut, L_thr))
            plt.semilogy(Ls / Db_Per_Km, Rs, style, lw=2.5, color=col,
                         label=(rf"$\delta_{{ph}}=\delta_{{pol}}={100 * d_pol:g}\%$"
                                if bmax > 1 else None))

    # positivity cutoff with AD
    print(f"\nSection 1: positivity cutoff with AD , rounded to the nearest km")
    for d_ph, d_pol, bmax, cut, L_thr in rows:
        if bmax == B_Max:
            km = int(np.floor(cut / Db_Per_Km + 0.5))
            print(f"delta_ph = {100 * d_ph:g}%, delta_pol = {100 * d_pol:g}%: "
                  f"cutoff = {cut:.3f} dB ({km} km)")

    # distances at SKR >= 1e-15
    print(f"\nSection 2: distances at SKR >= {SKR_THRESHOLD:.0e} bits/round, rounded to the "
          f"nearest km (gains from unrounded distances)")
    dist = {(d_ph, d_pol, bmax): L_thr / Db_Per_Km for d_ph, d_pol, bmax, cut, L_thr in rows}
    for d_ph, d_pol in Miss_allignment_cases:
        d0 = dist[(d_ph, d_pol, 1)]  # no AD, km (unrounded)
        d1 = dist[(d_ph, d_pol, B_Max)]  # AD, km (unrounded)
        print(f"delta_ph = {100 * d_ph:g}%, delta_pol = {100 * d_pol:g}%: "
              f"no AD = {d0 * Db_Per_Km:.3f} dB ({int(np.floor(d0 + 0.5))} km), "
              f"AD = {d1 * Db_Per_Km:.3f} dB ({int(np.floor(d1 + 0.5))} km), "
              f"gain = {int(np.floor(d1 - d0 + 0.5))} km")


    # key rate, block size and error rates at the reported distances
    print(f"\nSection 3: key rate, block size and error rates at the reported distances")
    print(f"{'misalign.':>10} {'protocol':>10} | {'point':>12} | {'km':>4} | {'rate':>9} | "
          f"{'b':>3} | {'mu':>9} | {'e_b':>6} | {'e_p^U':>6} | {'e_b+e_p^U':>9}")
    for d_ph, d_pol, bmax, cut, L_thr in rows:
        prot = "no AD" if bmax == 1 else f"AD b<={bmax}"
        for label, L in ((f"SKR>={SKR_THRESHOLD:.0e}", L_thr), ("positivity", cut)):
            R, mu, b, eb, ep = best_point(L, d_ph, d_pol, bmax)
            km = int(np.floor(L / Db_Per_Km + 0.5))
            print(f"{100 * d_ph:9g}% {prot:>10} | {label:>12} | {km:4d} | {R:9.2e} | "
                  f"{b:3d} | {mu:9.3e} | {eb:6.4f} | {ep:6.4f} | {eb + ep:9.4f}")

    # check: the no-AD rate drops from ~1e-11 to 0 within 0.005 dB (0.025 km)
    print("\nCheck (no AD): rate just beyond the reported 1e-15 distance")
    for d_ph, d_pol, bmax, cut, L_thr in rows:
        if bmax == 1:
            R_here = best_rate(L_thr, d_ph, d_pol, 1)
            R_next = best_rate(L_thr + 0.005, d_ph, d_pol, 1)
            print(f"{100 * d_ph:g}%: R = {R_here:.2e} at {L_thr / Db_Per_Km:.3f} km, "
                  f"R = {R_next:.2e} at {(L_thr + 0.005) / Db_Per_Km:.3f} km")


    L = np.linspace(0.001, 120, 800)
    plt.semilogy(L / Db_Per_Km, -np.log2(1 - 10 ** (-L / 10)), ":", color="0.4",
                 label=r"SKC$_0$")
    plt.axhline(SKR_THRESHOLD, color="0.6", lw=0.8, ls="-.")
    plt.xlim(0, 600);
    plt.ylim(1e-15, 1)
    plt.xlabel("Distance (km)");
    plt.ylabel("Secret-key rate (bits/round)")
    plt.title(f"solid: with AD (b <= {B_Max}), dashed: without AD")
    plt.grid(alpha=0.3);
    plt.legend()
    plt.tight_layout()
    plt.show()
