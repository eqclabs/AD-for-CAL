
# misalignment_tolerance.py for table 2 of the paper
# Table 2: largest tolerable misalignment at 1 km for SKR >= 1e-15 and 1e-21 bits/round,
# without AD (b = 1) and with AD (q = 0).
# 1e-7 <= mu <= 0.4 and block sizes 1 <= b <= 20000 (scan stopped once the rate bound pXX*pAD/b, which decreases with b, falls below R_Floor).

import numpy as np
from scipy.special import factorial, comb
from scipy.optimize import minimize_scalar
from concurrent.futures import ProcessPoolExecutor

#parameters (same names and values as the main script)
pd = 1e-8          # dark-count probability
fec = 1.16         # error-correction inefficiency
Ncut = 40          # truncation of cat-state sums
B_Max = 20000      # max AD block size for Table 2
R_Floor = 1e-30    # stop the b scan once pXX*pAD/b < R_Floor (far below the thresholds >= 1e-21)
Db_Per_Km = 0.2

Known_S = [(0, 0), (0, 2), (2, 0), (2, 2), (0, 4), (4, 0), (1, 1), (1, 3), (3, 1)]
SJ = {j: [((a - j) // 2, (b - j) // 2) for a, b in Known_S if a % 2 == j and b % 2 == j]
      for j in (0, 1)}

# intensity grid of our main script, extended down to 1e-7
Mu_Grid = np.unique(np.concatenate([np.geomspace(1e-7, 1e-4, 15),
                                    np.linspace(1e-4, 0.02, 25),
                                    np.linspace(0.02, 0.06, 25),
                                    np.linspace(0.06, 0.4, 20)]))

Targets = [(1.0, 1e-15), (1.0, 1e-21)]        # (distance in km, minimum SKR in bits/round)
Modes = ["both", "ph", "pol"]                  # simultaneous, phase only, polarisation only
Mode_Map = {"both": lambda d: (d, d), "ph": lambda d: (d, 0.0), "pol": lambda d: (0.0, d)}
Tol_Misalign = 1e-4                            # bisection accuracy in misalignment (0.01 %)
N_Workers = 4                                  # parallel processes; set to your number of CPU cores


#Equations (identical to the main script)

def h(x):
    x = np.asarray(x, dtype=float)
    xc = np.clip(x, 1e-300, 0.5)
    v = -xc * np.log2(xc) - (1 - xc) * np.log2(1 - xc)
    return np.where(x <= 0, 0.0, np.where(x >= 0.5, 1.0, v))

#tail bound
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
    ep = min(0.5, ep)  #equation 37
    if b == 1:
        pAD, eb2, ep2 = 1.0, eb, ep
    else:
        pAD = (1 - eb) ** b + eb ** b                 #equation 9
        eb2 = eb ** b / pAD         #equation 10
        ep2 = ((1 - eb) ** b - (1 - eb - 2 * ep) ** b) / (2 * pAD)  # ← q = 0  (Eq. (46), first branch)
        ep2 = min(max(ep2, 0.0), 0.5)
    bracket = 1 - fec * h(eb2) - h(ep2)
    return (pxx / b) * pAD * bracket if bracket >= 1e-12 else 0.0

def rate(mu, eta, cache, Om, bmax = B_Max):
    g = np.sqrt(eta) * mu
    ann = []
    for kc in (1, 0):                       # announcements (1,0) and (0,1)
        pxx = pXX(kc, g, Om)
        if pxx <= 0:
            return 0.0
        ann.append((pxx, eX(kc, g, Om), eZ_upp(mu, pxx, cache)))
    best = 0.0
    for b in range(1, bmax + 1):
        best = max(best, sum(rate_k(*a, b) for a in ann))
        # rate_k <= (pxx/b)*pAD, which only decreases with b: stop once it is below R_Floor
        if sum(a[0] * ((1 - a[1]) ** b + a[1] ** b) / b for a in ann) < R_Floor:
            break
    return best


# Optimum over mu (same steps as our best_rate of our main script)

def best_point(L, d_ph, d_pol, bmax = B_Max):
    eta = 10 ** (-L / 10)
    tA = np.arcsin(np.sqrt(d_pol)); tB = -tA
    Om = np.cos(d_ph * np.pi) * np.cos(tA - tB)
    cache = pzz_cache(eta, tA, tB)
    # error state only silences underflow warnings for large b (rate_k then returns 0)
    with np.errstate(all="ignore"):
        vals = [rate(mu, eta, cache, Om, bmax) for mu in Mu_Grid]
        i = int(np.argmax(vals))
        lo, hi = Mu_Grid[max(i - 1, 0)], Mu_Grid[min(i + 1, len(Mu_Grid) - 1)]
        res = minimize_scalar(lambda mu: -rate(mu, eta, cache, Om, bmax),
                              bounds=(lo, hi), method="bounded",
                              options={"xatol": max(1e-13, 1e-4 * lo)})
        R, mu = (-res.fun, res.x) if -res.fun > vals[i] else (vals[i], Mu_Grid[i])

        # block size and error rates at the optimum (announcement (1,0); (0,1) is identical)
        g = np.sqrt(eta) * mu
        ann = [(pXX(kc, g, Om), eX(kc, g, Om), eZ_upp(mu, pXX(kc, g, Om), cache))
               for kc in (1, 0)]
        Rb = []
        for bb in range(1, bmax + 1):
            Rb.append(sum(rate_k(*a, bb) for a in ann))
            if sum(a[0] * ((1 - a[1]) ** bb + a[1] ** bb) / bb for a in ann) < R_Floor:
                break
        b = int(np.argmax(Rb)) + 1 if R > 0 else 0
    return dict(R=R, mu=mu, b=b, eb=ann[0][1], ep=min(0.5, ann[0][2]))


# Largest tolerable misalignment (bisection in delta)

def max_tolerance(task):
    km, SKR_min, mode, ad = task
    L = km * Db_Per_Km
    bmax = B_Max if ad else 1
    ok = lambda p: p["R"] >= SKR_min

    def at(d):
        d_ph, d_pol = Mode_Map[mode](d)
        return best_point(L, d_ph, d_pol, bmax)

    lo, hi = 0.0, 0.5
    p_lo = at(lo)
    if not ok(p_lo):
        return task, None, p_lo
    p_hi = at(hi)
    if ok(p_hi):
        return task, hi, p_hi
    while hi - lo > Tol_Misalign:
        mid = 0.5 * (lo + hi)
        p = at(mid)
        if ok(p):
            lo, p_lo = mid, p
        else:
            hi = mid
    return task, lo, p_lo


# Output

Mode_Name = {"both": "phase + pol.", "ph": "phase only", "pol": "pol. only"}

def print_results(out):
    print(f"\nf = {fec}, pd = {pd:g}, q = 0, b = 1..{B_Max} (scan stopped below {R_Floor:g}), "
          f"mu in [{Mu_Grid.min():g}, {Mu_Grid.max():g}] ({len(Mu_Grid)} grid points + Brent), "
          f"accuracy {100 * Tol_Misalign:g} %")
    all_ok = True
    for km, SKR_min in Targets:
        print("\n" + "=" * 112)
        print(f"Distance {km:g} km ({km * Db_Per_Km:g} dB), SKR >= {SKR_min:g} bits/round")
        print("-" * 112)
        print(f"{'misalignment':>13} {'AD':>4} | {'delta_max':>9} | {'b':>5} {'mu':>10} "
              f"{'rate at edge':>12} {'e_b':>7} {'e_p^U':>7} {'e_b+e_p^U':>10} {'q*':>8}  notes")
        for (k, rm, mode, ad), d, p in out:
            if (k, rm) != (km, SKR_min):
                continue
            tol = f"{100 * d:8.2f}%" if d is not None else "     none"
            line = f"{Mode_Name[mode]:>13} {'yes' if ad else 'no':>4} | {tol:>9} | "
            notes = []
            if p["b"] > 0:
                s = p["eb"] + p["ep"]   # e_b + e_p^U at the optimum
                qs = s / 2 - 0.25      # q* = (e_b + e_p^U − 1/2) / 2, reason for taking q=0 alaways
                all_ok &= (s <= 0.5)
                line += (f"{p['b']:5d} {p['mu']:10.3e} {p['R']:12.2e} "
                         f"{p['eb']:7.4f} {p['ep']:7.4f} {s:10.4f} {qs:+8.4f}")
                if ad and p["b"] >= B_Max:
                    notes.append("b at top of range")
                if p["mu"] <= Mu_Grid[0] * 1.0001:
                    notes.append("mu at bottom of grid")
                if p["mu"] >= Mu_Grid[-1] * 0.9999:
                    notes.append("mu at top of grid")
            if d is not None and d >= 0.5:
                notes.append("tolerable up to 50 % (search limit)")
            print(line + ("  " + "; ".join(notes) if notes else ""))
    print(f"\nAll optimising points satisfy e_b + e_p^U <= 1/2 (q* <= 0): {all_ok}")

def print_table2(out):
    res = {(k, rm, mode, ad): (d, p) for (k, rm, mode, ad), d, p in out}
    print("\n" + "=" * 80)
    print("TABLE 2")
    print("-" * 80)
    print(f"{'Distance':>8} {'SKR_min':>8} {'Misalignment':>13} | "
          f"{'no AD (%)':>9} {'AD (%)':>7} {'b':>5} {'mu':>10}")
    for km, SKR_min in Targets:
        for mode in Modes:
            d0, _ = res[(km, SKR_min, mode, False)]
            d1, p1 = res[(km, SKR_min, mode, True)]
            f = lambda d: f"{100 * d:.1f}" if d is not None else "none"
            b = p1["b"] if p1["b"] > 0 else "-"
            mu = f"{p1['mu']:.1e}" if p1["b"] > 0 else "-"
            print(f"{km:>6g}km {SKR_min:>8.0e} {Mode_Name[mode]:>13} | "
                  f"{f(d0):>9} {f(d1):>7} {b:>5} {mu:>10}")

    
    # check: without AD the rate drops from ~1e-8 to 0 within one bisection step (0.01 %)
    print("\nCheck (no AD): rate at delta_max and one bisection step (0.01 %) beyond it")
    done = set()
    for (km, SKR_min, mode, ad), d, p in out:
        if ad or d is None or (km, mode) in done:
            continue
        done.add((km, mode))                       # no-AD delta_max is the same for both thresholds
        d_ph, d_pol = Mode_Map[mode](d + Tol_Misalign)
        R_next = best_point(km * Db_Per_Km, d_ph, d_pol, 1)["R"]
        print(f"{Mode_Name[mode]:>13}: R = {p['R']:.2e} at delta = {100 * d:.2f} %, "
              f"R = {R_next:.2e} at delta = {100 * (d + Tol_Misalign):.2f} %")
# Main

if __name__ == "__main__":
    tasks = [(km, SKR_min, mode, ad)
             for (km, SKR_min) in Targets for mode in Modes for ad in (False, True)]
    workers = min(len(tasks), N_Workers)
    print(f"{len(tasks)} cases, {workers} worker(s) ... (b up to {B_Max}, "
          f"scan stopped once pXX*pAD/b < {R_Floor:g})")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(max_tolerance, tasks))
    else:
        out = [max_tolerance(t) for t in tasks]
    print_results(out)
    print_table2(out)
