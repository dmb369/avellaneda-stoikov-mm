"""
Avellaneda-Stoikov market making backtest on Yahoo Finance OHLCV bars (yfinance).

Install : pip install yfinance pandas numpy scipy matplotlib
Run     : python av_yf.py                                  # BTC-USD,ETH-USD,SOL-USD on 5m bars
          python av_yf.py BTC-USD,ETH-USD,SOL-USD,XRP-USD 5m
          python av_yf.py BTC-USD 1m                      # ~4 weeks of 1-minute bars
          python av_yf.py BTC-USD 60m                     # ~2 years of hourly bars
          python av_yf.py SPY,QQQ,AAPL,MSFT,NVDA 5m       # US stocks / ETFs
          python av_yf.py ES=F,GC=F,CL=F 5m               # futures
          python av_yf.py EURUSD=X,GBPUSD=X 5m            # FX (Yahoo gives no FX volume; handled)
Session length and the annualisation factor adapt automatically to exchange-hours assets.

Yahoo limits (as of writing): 1m ~30 days back (fetched in 7-day chunks), 2m-30m ~60 days,
60m ~730 days. Data is cached in ./yf_cache; delete the folder to re-download.

HOW BARS STAND IN FOR A TAPE
* mid proxy        = previous bar's close
* a bid at b fills in a bar if the bar's LOW  < b - haircut   (a trade printed through it)
* an ask at a fills in a bar if the bar's HIGH > a + haircut
* haircut (bps)    = crude queue-position penalty; swept in Experiment 4
* one unit per side per bar, hard inventory cap, leftover inventory unwound at
  half-spread + slippage at the end of each session
* sigma, kappa are calibrated on the CAL_DAYS days BEFORE each session (out-of-sample)

WHAT THIS CAN AND CANNOT TELL YOU
Bar high/low are far wider than the instantaneous spread, so fill rates and edge per fill are
much more generous than on a real tape. Treat absolute PnL as an upper bound and focus on the
RELATIVE comparisons (AS vs constant-skew vs fixed, and how they react to fees, haircut and
latency). Do not describe this as tick-level or order-book-level data.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
TICKERS = (sys.argv[1] if len(sys.argv) > 1 else "BTC-USD,ETH-USD,SOL-USD").split(",")
INTERVAL = sys.argv[2] if len(sys.argv) > 2 else "5m"
CACHE_DIR = "yf_cache"

_SEC = {"1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600, "1h": 3600}
_LOOKBACK = {"1m": 28, "2m": 58, "5m": 58, "15m": 58, "30m": 58, "60m": 700, "1h": 700}
_CALDAYS = {"1m": 3, "2m": 4, "5m": 5, "15m": 7, "30m": 10, "60m": 15, "1h": 15}
INTERVAL_SEC = _SEC[INTERVAL]
LOOKBACK_DAYS = _LOOKBACK[INTERVAL]
CAL_DAYS = _CALDAYS[INTERVAL]

DEFAULT_SESSION_BARS = 48   # 5m -> 4h, 1m -> 48min, 60m -> 2 days (auto-shortened for exchange-hours assets)
SESSION_BARS = DEFAULT_SESSION_BARS
LOT_USD = 1000.0            # notional of ONE inventory unit
MAX_INV = 25                # hard inventory cap in units
SKEW_SIGMA = 0.05           # AS skew per unit at session start, as a fraction of per-bar sigma
SKEW_SWEEP = [0.01, 0.025, 0.05, 0.1, 0.2]
NAIVE_MULTS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]    # multiples of the matched half-spread
SIGMA_BARS = 3              # sigma from 3-bar returns, rescaled to 1 bar
MARKOUT_BARS = 2
LIQ_BPS = 1.0               # extra unwind slippage (bps of price) beyond the half-spread
FEE_BPS_LEVELS = [-1.0, 0.0, 0.5, 1.0, 2.0]
HAIRCUT_SWEEP_BPS = [0.0, 1.0, 2.0, 4.0]
LATENCY_SWEEP = [0, 1, 2]   # bars of staleness in the mid the quoter sees
DELTAS_BPS = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80], float)
ANN_DAYS = 365              # set automatically: 365 for 24/7 assets, 252 otherwise
SEED = 42


# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
def fetch(ticker):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{ticker}_{INTERVAL}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    import yfinance as yf
    end = pd.Timestamp.now(tz="UTC").floor("D") + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    if INTERVAL == "1m":                      # Yahoo allows max 7 days per 1m request
        edges = list(pd.date_range(start, end, freq="7D", tz="UTC")) + [end]
        windows = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    else:
        windows = [(start, end)]

    frames = []
    for a, b in windows:
        d = yf.download(ticker, start=a.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                        interval=INTERVAL, auto_adjust=False, progress=False)
        if d is None or len(d) == 0:
            continue
        if isinstance(d.columns, pd.MultiIndex):        # newer yfinance returns (field, ticker)
            d.columns = d.columns.get_level_values(0)
        frames.append(d[["Open", "High", "Low", "Close", "Volume"]])
    if not frames:
        raise SystemExit(f"No data returned for {ticker} at interval {INTERVAL}")
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df = df.dropna()
    df = df[(df["High"] >= df["Low"]) & (df["Low"] > 0)]
    df = df.iloc[:-1]                                   # last bar may still be forming
    df.to_csv(path)
    return df


def prepare(df):
    ts = ((df.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy().astype("int64")
    o, h, l, c, v = (df[k].to_numpy(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    cont = np.zeros(len(df), bool)
    cont[1:] = np.diff(ts) == INTERVAL_SEC              # bar follows its predecessor with no gap
    pre = o.copy()
    pre[1:] = np.where(cont[1:], c[:-1], o[1:])         # mid known at the START of each bar
    day = ts // 86400
    days = np.unique(day)
    start = np.searchsorted(day, days)
    return {"has_vol": bool(v.sum() > 0), "ts": ts, "o": o, "h": h, "l": l, "c": c, "v": v, "cont": cont, "pre": pre,
            "day": day, "days": days, "start": start, "n": len(df)}


# ----------------------------------------------------------------------
# 2. Calibration on a trailing window of bars [a, b)
# ----------------------------------------------------------------------
def calibrate(B, a, b):
    i = np.arange(a + 1, b)
    i = i[B["cont"][i]]
    if len(i) < 100:
        return None
    ref, hi, lo = B["c"][i - 1], B["h"][i], B["l"][i]
    px = float(np.median(B["c"][a:b]))
    deltas = DELTAS_BPS * 1e-4 * px
    p = 0.5 * ((lo[:, None] < ref[:, None] - deltas).mean(0) +
               (hi[:, None] > ref[:, None] + deltas).mean(0))
    lam = -np.log1p(-np.clip(p, 0, 0.999))              # fills per bar at distance delta
    ok = (p > 0.005) & (p < 0.9)
    if ok.sum() < 3:
        return None
    x, y = deltas[ok], np.log(lam[ok])
    coef = np.polyfit(x, y, 1)
    kappa = -coef[0]
    if not np.isfinite(kappa) or kappa <= 0:
        return None
    r2 = 1 - ((y - np.polyval(coef, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    hh = SIGMA_BARS
    j = np.arange(a + hh, b)
    j = j[(B["ts"][j] - B["ts"][j - hh]) == hh * INTERVAL_SEC]
    if len(j) < 50:
        return None
    sigma = float(np.std(B["c"][j] - B["c"][j - hh]) / np.sqrt(hh))
    return {"sigma": sigma, "kappa": float(kappa), "A": float(np.exp(coef[1])), "r2": float(r2),
            "px": px, "deltas": deltas, "lam": lam, "ok": ok}


# ----------------------------------------------------------------------
# 3. Quoting strategies ($ per unit of the asset; one inventory unit = LOT_USD)
# ----------------------------------------------------------------------
def gamma_from_g(g, sigma, n=None):
    n = SESSION_BARS if n is None else n
    return g / (sigma ** 2 * n)             # g = gamma * sigma^2 * T = $ skew per unit at start


def as_avg_half_spread(g, gamma, kappa):
    return 0.5 * (g / 2 + (2 / gamma) * np.log1p(gamma / kappa))


def make_as(g, gamma, kappa):
    def quote(m, q, tau):
        res = m - q * g * tau
        half = 0.5 * (g * tau + (2 / gamma) * np.log1p(gamma / kappa))
        return res - half, res + half
    return quote


def make_skew(g, half):
    """Constant skew, no time decay (infinite-horizon style): AS's natural simple competitor."""
    def quote(m, q, tau):
        res = m - q * g
        return res - half, res + half
    return quote


def make_naive(half):
    return lambda m, q, tau: (m - half, m + half)


# ----------------------------------------------------------------------
# 4. Sessions and simulator (vectorised across sessions)
# ----------------------------------------------------------------------
def build_sessions(B):
    cache, rows = {}, []
    pre, hi, lo, post = [], [], [], []
    first = B["start"][CAL_DAYS] if len(B["days"]) > CAL_DAYS else B["n"]
    i = first
    while i + SESSION_BARS <= B["n"]:
        seg = slice(i, i + SESSION_BARS)
        good = B["cont"][i + 1:i + SESSION_BARS].all() and (not B["has_vol"] or (B["v"][seg] > 0).mean() >= 0.5)
        if not good:
            i += 1
            continue
        k = int(np.searchsorted(B["days"], B["day"][i]))
        if k < CAL_DAYS:
            i += 1
            continue
        if k not in cache:
            cache[k] = calibrate(B, int(B["start"][k - CAL_DAYS]), int(B["start"][k]))
        cal = cache[k]
        if cal is None:
            i += SESSION_BARS
            continue
        pre.append(B["pre"][seg]); hi.append(B["h"][seg]); lo.append(B["l"][seg]); post.append(B["c"][seg])
        rows.append((str(pd.Timestamp(B["ts"][i], unit="s").date()), cal["sigma"], cal["kappa"], cal["px"]))
        i += SESSION_BARS
    if not rows:
        raise SystemExit("No usable sessions (data too short or too many gaps).")
    meta = pd.DataFrame(rows, columns=["date", "sigma", "kappa", "px"])
    S = {"pre": np.array(pre).T.copy(), "hi": np.array(hi).T.copy(),
         "lo": np.array(lo).T.copy(), "post": np.array(post).T.copy(),
         "px": meta["px"].to_numpy(), "lot": LOT_USD / meta["px"].to_numpy()}
    cals = [c for _, c in sorted(cache.items()) if c is not None]
    return S, meta, cals


def simulate(S, quoter, liq, haircut_bps=0.0, latency=0, max_inv=MAX_INV):
    N, M = S["pre"].shape
    pen = haircut_bps * 1e-4 * S["px"]
    q = np.zeros(M); cash = np.zeros(M); notional = np.zeros(M); fills = np.zeros(M)
    edge = np.zeros(M); markout = np.zeros(M); capped = np.zeros(M)
    peak = np.zeros(M); dd = np.zeros(M); sum_q2 = np.zeros(M); max_q = np.zeros(M)
    for t in range(N):
        tau = 1.0 - t / N
        bid, ask = quoter(S["pre"][max(t - latency, 0)], q, tau)
        live_b, live_a = q < max_inv, q > -max_inv
        capped += ~(live_b & live_a)
        hb = (S["lo"][t] < bid - pen) & live_b
        ha = (S["hi"][t] > ask + pen) & live_a
        q += hb; q -= ha
        cash += np.where(hb, -bid, 0.0) + np.where(ha, ask, 0.0)
        notional += np.where(hb, bid, 0.0) + np.where(ha, ask, 0.0)
        fills += hb.astype(float) + ha                    # NB: bool+bool would be a logical OR
        mid = S["pre"][t]
        mh = S["post"][min(t + MARKOUT_BARS, N - 1)]
        edge += np.where(hb, (mid - bid) / mid * 1e4, 0.0) + np.where(ha, (ask - mid) / mid * 1e4, 0.0)
        markout += np.where(hb, (mh - bid) / mid * 1e4, 0.0) + np.where(ha, (ask - mh) / mid * 1e4, 0.0)
        pnl_t = cash + q * S["post"][t] - liq * np.abs(q)
        peak = np.maximum(peak, pnl_t)
        dd = np.minimum(dd, pnl_t - peak)
        sum_q2 += q * q
        max_q = np.maximum(max_q, np.abs(q))
    final = cash + q * S["post"][-1] - liq * np.abs(q)
    safe = np.where(fills > 0, fills, np.nan)
    return {"pnl": final * S["lot"], "dd": dd * S["lot"], "rms_inv": np.sqrt(sum_q2 / N),
            "max_inv": max_q, "fills": fills, "notional": notional * S["lot"],
            "edge_bps": edge / safe, "markout_bps": markout / safe, "capped_frac": capped / N}


# ----------------------------------------------------------------------
# 5. Stats helpers
# ----------------------------------------------------------------------
def sharpe(x):
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else np.nan


def col_mean(x):
    return float(np.nanmean(x))


def cluster_ci(fn, a, b, day_idx, rng, n_boot=2000):
    groups = [np.where(day_idx == d)[0] for d in np.unique(day_idx)]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        out.append(fn(a[idx], b[idx]))
    return np.nanpercentile(out, [2.5, 97.5])


# ----------------------------------------------------------------------
# 6. One ticker end to end
# ----------------------------------------------------------------------
def run(ticker):
    global SESSION_BARS, ANN_DAYS
    rng = np.random.default_rng(SEED)
    df = fetch(ticker)
    B = prepare(df)
    SESSION_BARS, ANN_DAYS = DEFAULT_SESSION_BARS, 365
    bars_per_day = len(df) / max(len(B["days"]), 1)
    if bars_per_day < 0.9 * 86400 / INTERVAL_SEC:          # exchange hours, not 24/7
        SESSION_BARS = int(max(12, min(DEFAULT_SESSION_BARS, bars_per_day // 3)))
        ANN_DAYS = 252
    print(f"{ticker}: {len(df):,} bars of {INTERVAL}, {len(B['days'])} days "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    if len(B["days"]) <= CAL_DAYS + 3:
        raise SystemExit(f"{ticker}: need more than {CAL_DAYS + 3} days of data")

    S, meta, cals = build_sessions(B)
    day_idx = pd.factorize(meta["date"])[0]
    n_days, n_sess = len(set(day_idx)), len(meta)
    sigma, kappa, px = (meta[k].to_numpy() for k in ["sigma", "kappa", "px"])
    print(f"Sessions: {n_sess} across {n_days} trading days "
          f"(each {SESSION_BARS} bars = {SESSION_BARS * INTERVAL_SEC / 3600:.1f}h, "
          f"calibrated on the prior {CAL_DAYS} days)")
    print(f"Median calibration: sigma {np.median(sigma / px) * 1e4:.1f} bps/bar, "
          f"kappa {np.median(kappa * px) / 1e4:.2f} per bps, fit R^2 {np.median([c['r2'] for c in cals]):.2f}")

    g0 = SKEW_SIGMA * sigma
    gamma = gamma_from_g(g0, sigma)
    half = as_avg_half_spread(g0, gamma, kappa)
    liq = half + LIQ_BPS * 1e-4 * px
    print(f"Avg half-spread {np.mean(half / px) * 1e4:.2f} bps | skew/unit at start {SKEW_SIGMA} x sigma")

    R_as = simulate(S, make_as(g0, gamma, kappa), liq)
    R_ls = simulate(S, make_skew(g0 / 2, half), liq)
    R_nv = simulate(S, make_naive(half), liq)
    Rs = {"AS": R_as, "Const-skew": R_ls, "Fixed": R_nv}

    # ---- Experiment 1 ---------------------------------------------------
    print("\n" + "=" * 78)
    print("Experiment 1: AS vs constant-skew vs fixed spread (equal average width, fee = 0)")
    print("Const-skew has the SAME average skew as AS but no time decay.")
    print("=" * 78)

    def metric(R, key):
        if key == "std": return R["pnl"].std(ddof=1)
        if key == "sharpe": return sharpe(R["pnl"])
        if key == "adv": return col_mean(R["edge_bps"]) - col_mean(R["markout_bps"])
        return col_mean(R[key])

    rows = [("Mean PnL / session ($)", "pnl"), ("PnL std across sessions ($)", "std"),
            ("Sharpe (per session)", "sharpe"), ("Avg max drawdown ($)", "dd"),
            ("RMS inventory (units)", "rms_inv"), ("Max |inventory| (units)", "max_inv"),
            ("Fraction of time at cap", "capped_frac"), ("Fills / session", "fills"),
            ("Quoted edge per fill (bps)", "edge_bps"),
            (f"{MARKOUT_BARS}-bar markout per fill (bps)", "markout_bps"),
            ("Adverse selection (bps)", "adv")]
    print(f"{'Metric':<34}" + "".join(f"{k:>14}" for k in Rs))
    for label, key in rows:
        print(f"{label:<34}" + "".join(f"{metric(R, key):>14.3f}" for R in Rs.values()))

    daily = pd.DataFrame({"d": meta["date"], "as": R_as["pnl"], "ls": R_ls["pnl"],
                          "nv": R_nv["pnl"]}).groupby("d").sum()
    ann = {k: (daily[k].mean() / daily[k].std(ddof=1) * np.sqrt(ANN_DAYS)
               if len(daily) > 2 and daily[k].std(ddof=1) > 0 else np.nan) for k in ["as", "ls", "nv"]}
    print(f"{'Daily Sharpe (annualised)':<34}{ann['as']:>14.2f}{ann['ls']:>14.2f}{ann['nv']:>14.2f}")

    print("-" * 78)
    for name, R_b in [("fixed", R_nv), ("const-skew", R_ls)]:
        diff = R_as["pnl"] - R_b["pnl"]
        red = 1 - R_as["rms_inv"].mean() / R_b["rms_inv"].mean()
        print(f"AS vs {name}: RMS-inventory reduction {100 * red:.1f}%")
        if np.allclose(diff, 0):
            print("  PnL identical (no fills?)")
            continue
        tp = stats.ttest_rel(R_as["pnl"], R_b["pnl"])[1]
        wp = stats.wilcoxon(R_as["pnl"], R_b["pnl"])[1]
        print(f"  mean PnL diff ${diff.mean():+.4f} (paired t p={tp:.4f}) | "
              f"median ${np.median(diff):+.4f} (Wilcoxon p={wp:.4f})")
        if n_days >= 3:
            lo_, hi_ = cluster_ci(lambda x, y: 1 - x.mean() / y.mean(), R_as["rms_inv"], R_b["rms_inv"], day_idx, rng)
            print(f"  RMS reduction 95% CI (day bootstrap): [{100 * lo_:.1f}%, {100 * hi_:.1f}%]")
            lo_, hi_ = cluster_ci(lambda x, y: x.mean() - y.mean(), R_as["pnl"], R_b["pnl"], day_idx, rng)
            print(f"  mean PnL diff 95% CI: [${lo_:+.3f}, ${hi_:+.3f}]")
            lo_, hi_ = cluster_ci(lambda x, y: sharpe(x) - sharpe(y), R_as["pnl"], R_b["pnl"], day_idx, rng)
            print(f"  Sharpe diff 95% CI:   [{lo_:+.3f}, {hi_:+.3f}]  (positive = AS better risk-adjusted)")
            lo_, hi_ = cluster_ci(lambda x, y: x.std(ddof=1) / y.std(ddof=1), R_as["pnl"], R_b["pnl"], day_idx, rng)
            print(f"  PnL std ratio AS/{name}: {R_as['pnl'].std(ddof=1) / R_b['pnl'].std(ddof=1):.3f} "
                  f"(95% CI {lo_:.3f} to {hi_:.3f})")

    # ---- Experiment 2: fees ---------------------------------------------
    print("\n" + "=" * 78)
    print("Experiment 2: maker-fee sensitivity (mean net PnL / session, $)")
    print("=" * 78)
    print(f"{'fee (bps)':<12}{'AS':>10}{'Const-skew':>12}{'Fixed':>10}")
    fee_rows = []
    for fee in FEE_BPS_LEVELS:
        nets = [(R["pnl"] - fee * 1e-4 * R["notional"]).mean() for R in Rs.values()]
        fee_rows.append((fee, *nets))
        print(f"{fee:<12.1f}{nets[0]:>10.3f}{nets[1]:>12.3f}{nets[2]:>10.3f}")
    be = {k: 1e4 * R["pnl"].sum() / R["notional"].sum() for k, R in Rs.items()}
    print(f"Break-even maker fee (bps): AS {be['AS']:.3f} | Const-skew {be['Const-skew']:.3f} | "
          f"Fixed {be['Fixed']:.3f}   <- compare with your real fee tier")

    # ---- Experiment 3: frontier -----------------------------------------
    print("\n" + "=" * 78)
    print("Experiment 3: risk-return frontier (mean PnL, PnL std, $ per session)")
    print("=" * 78)
    as_front, nv_front = [], []
    for c in SKEW_SWEEP:
        g = c * sigma
        gm = gamma_from_g(g, sigma)
        R = simulate(S, make_as(g, gm, kappa), half + LIQ_BPS * 1e-4 * px)
        as_front.append((c, R["pnl"].mean(), R["pnl"].std(ddof=1)))
    for mlt in NAIVE_MULTS:
        R = simulate(S, make_naive(mlt * half), liq)
        nv_front.append((mlt, R["pnl"].mean(), R["pnl"].std(ddof=1)))
    print("AS    (skew x sigma, mean, std):", [(c, round(float(m), 3), round(float(s), 3)) for c, m, s in as_front])
    print("Fixed (x matched half, mean, std):", [(k, round(float(m), 3), round(float(s), 3)) for k, m, s in nv_front])
    srt = sorted((s_, m_) for _, m_, s_ in nv_front)
    xs_, ys_ = [x for x, _ in srt], [y for _, y in srt]
    as_std, as_mean = R_as["pnl"].std(ddof=1), R_as["pnl"].mean()
    fixed_at = np.nan
    if xs_[0] <= as_std <= xs_[-1]:
        fixed_at = float(np.interp(as_std, xs_, ys_))
        print(f"Matched risk: at PnL std ${as_std:.3f} AS earns ${as_mean:.3f}; a fixed spread tuned to the "
              f"same std earns ~${fixed_at:.3f} (AS edge ${as_mean - fixed_at:+.3f})")
    else:
        print(f"Matched risk: AS std ${as_std:.3f} is outside the fixed frontier "
              f"[${xs_[0]:.3f}, ${xs_[-1]:.3f}]; compare on Sharpe instead.")

    # ---- Experiment 4: fill haircut -------------------------------------
    print("\n" + "=" * 78)
    print("Experiment 4: fill haircut -- bar must trade THROUGH the quote by this many bps")
    print("(the bar-data stand-in for queue position; the most important realism check)")
    print("=" * 78)
    print(f"{'haircut (bps)':<15}{'AS mean':>10}{'AS Sharpe':>11}{'Skew mean':>11}{'Fixed mean':>12}{'AS fills':>10}")
    for hc in HAIRCUT_SWEEP_BPS:
        Ra = simulate(S, make_as(g0, gamma, kappa), liq, haircut_bps=hc)
        Rl = simulate(S, make_skew(g0 / 2, half), liq, haircut_bps=hc)
        Rn = simulate(S, make_naive(half), liq, haircut_bps=hc)
        print(f"{hc:<15.1f}{Ra['pnl'].mean():>10.3f}{sharpe(Ra['pnl']):>11.3f}{Rl['pnl'].mean():>11.3f}"
              f"{Rn['pnl'].mean():>12.3f}{Ra['fills'].mean():>10.1f}")

    # ---- Experiment 5: latency ------------------------------------------
    print("\n" + "=" * 78)
    print("Experiment 5: quote latency (quoter sees a mid that is L bars stale)")
    print("=" * 78)
    print(f"{'latency (bars)':<16}{'AS mean':>10}{'AS std':>9}{'Skew mean':>11}{'Fixed mean':>12}{'Fixed std':>11}")
    for L in LATENCY_SWEEP:
        Ra = simulate(S, make_as(g0, gamma, kappa), liq, latency=L)
        Rl = simulate(S, make_skew(g0 / 2, half), liq, latency=L)
        Rn = simulate(S, make_naive(half), liq, latency=L)
        print(f"{L:<16}{Ra['pnl'].mean():>10.3f}{Ra['pnl'].std(ddof=1):>9.3f}{Rl['pnl'].mean():>11.3f}"
              f"{Rn['pnl'].mean():>12.3f}{Rn['pnl'].std(ddof=1):>11.3f}")

    # ---- Experiment 6: regimes ------------------------------------------
    print("\n" + "=" * 78)
    print("Experiment 6: volatility regimes (terciles of calibrated sigma / price)")
    print("=" * 78)
    rv = sigma / px
    q1, q2 = np.quantile(rv, [1 / 3, 2 / 3])
    print(f"{'regime':<10}{'n':>5}{'AS mean':>10}{'Fix mean':>10}{'AS std':>9}{'Fix std':>9}{'RMS red.':>10}")
    for name, mask in [("low vol", rv <= q1), ("mid vol", (rv > q1) & (rv <= q2)), ("high vol", rv > q2)]:
        if mask.sum() < 10:
            continue
        pa, pn = R_as["pnl"][mask], R_nv["pnl"][mask]
        red = 1 - R_as["rms_inv"][mask].mean() / R_nv["rms_inv"][mask].mean()
        print(f"{name:<10}{mask.sum():>5}{pa.mean():>10.3f}{pn.mean():>10.3f}"
              f"{pa.std(ddof=1):>9.3f}{pn.std(ddof=1):>9.3f}{100 * red:>9.1f}%")

    # ---- Outputs --------------------------------------------------------
    out = meta.copy()
    out["as_pnl"], out["skew_pnl"], out["fixed_pnl"] = R_as["pnl"], R_ls["pnl"], R_nv["pnl"]
    out["as_rms_inv"], out["skew_rms_inv"], out["fixed_rms_inv"] = R_as["rms_inv"], R_ls["rms_inv"], R_nv["rms_inv"]
    out.to_csv(f"{ticker}_{INTERVAL}_sessions.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    c0 = cals[-1]
    ax[0, 0].semilogy(c0["deltas"] / c0["px"] * 1e4, np.maximum(c0["lam"], 1e-6), "o", label="measured")
    ax[0, 0].semilogy(c0["deltas"] / c0["px"] * 1e4, c0["A"] * np.exp(-c0["kappa"] * c0["deltas"]), "-", label="exp fit")
    ax[0, 0].set_xlabel("Distance from mid (bps)"); ax[0, 0].set_ylabel("Fills per bar")
    ax[0, 0].set_title(f"{ticker} fill rate vs distance (R2={c0['r2']:.2f})"); ax[0, 0].legend()
    ax[0, 1].plot([s for _, _, s in as_front], [m for _, m, _ in as_front], "o-", label="AS (varying skew)")
    ax[0, 1].plot([s for _, _, s in nv_front], [m for _, m, _ in nv_front], "s-", label="Fixed (varying width)")
    ax[0, 1].scatter([R_ls["pnl"].std(ddof=1)], [R_ls["pnl"].mean()], marker="^", s=60, c="k", label="Const-skew", zorder=5)
    ax[0, 1].set_xlabel("PnL std across sessions ($)"); ax[0, 1].set_ylabel("Mean PnL ($)")
    ax[0, 1].set_title("Risk-return frontier"); ax[0, 1].legend()
    for k, lab in [("as_pnl", "AS"), ("skew_pnl", "Const-skew"), ("fixed_pnl", "Fixed")]:
        ax[1, 0].plot(np.cumsum(out[k].to_numpy()), label=lab)
    ax[1, 0].set_xlabel("Session #"); ax[1, 0].set_ylabel("Cumulative PnL ($)")
    ax[1, 0].set_title("Cumulative PnL (fee = 0)"); ax[1, 0].legend()
    for j, lab in [(1, "AS"), (2, "Const-skew"), (3, "Fixed")]:
        ax[1, 1].plot([r[0] for r in fee_rows], [r[j] for r in fee_rows], "o-", label=lab)
    ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[1, 1].set_xlabel("Maker fee (bps)"); ax[1, 1].set_ylabel("Mean net PnL / session ($)")
    ax[1, 1].set_title("Fee sensitivity"); ax[1, 1].legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_{INTERVAL}_results.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {ticker}_{INTERVAL}_results.png and {ticker}_{INTERVAL}_sessions.csv")

    return {"ticker": ticker, "sessions": n_sess, "days": n_days,
            "rms_red_vs_fixed_%": 100 * (1 - R_as["rms_inv"].mean() / R_nv["rms_inv"].mean()),
            "rms_red_vs_skew_%": 100 * (1 - R_as["rms_inv"].mean() / R_ls["rms_inv"].mean()),
            "as_pnl": R_as["pnl"].mean(), "skew_pnl": R_ls["pnl"].mean(), "fixed_pnl": R_nv["pnl"].mean(),
            "as_sharpe": sharpe(R_as["pnl"]), "skew_sharpe": sharpe(R_ls["pnl"]), "fixed_sharpe": sharpe(R_nv["pnl"]),
            "as_breakeven_fee_bps": be["AS"]}


def main():
    rows = []
    for t in TICKERS:
        print("\n" + "#" * 78 + f"\n# {t}  ({INTERVAL} bars)\n" + "#" * 78)
        try:
            rows.append(run(t))
        except SystemExit as e:
            print(f"Skipping {t}: {e}")
    if len(rows) > 1:
        summ = pd.DataFrame(rows)
        print("\n" + "=" * 78 + "\nCross-asset summary\n" + "=" * 78)
        print(summ.round(3).to_string(index=False))
        summ.to_csv(f"cross_asset_summary_{INTERVAL}.csv", index=False)


if __name__ == "__main__":
    main()
