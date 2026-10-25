"""
Avellaneda-Stoikov market making on REAL Binance BTCUSDT futures trades.

Input : daily trade files named like BTCUSDT-trades-2026-09-01.csv (or .zip)
        columns: id, price, qty, quote_qty, time, is_buyer_maker
Run   : python av2.py path/to/folder BTCUSDT,ETHUSDT,SOLUSDT
        (second argument optional, default BTCUSDT; each symbol needs its own daily files)

Question: on real tape, does inventory-aware quoting (AS) reduce risk versus a
fixed spread of the SAME average width, and what do fees and adverse selection do?

How the data is used
--------------------
* is_buyer_maker = True  -> a seller hit a resting BID  -> can fill our bid
* is_buyer_maker = False -> a buyer lifted a resting ASK -> can fill our ask
* Mid-price proxy = last trade price each second (no quote data needed).
* Fill rule ("through"): our bid at price b fills in a second only if a
  sell-side trade prints strictly BELOW b (ask: buy-side trade strictly above).
  Queue position is unknown, so requiring the price to trade through is the
  conservative choice. "touch" (<=) is reported as an optimistic sensitivity.
* Parameters (sigma, kappa, A) are calibrated on the PREVIOUS day and traded on
  the next day, so the backtest is out-of-sample.
* Sessions are 1 hour long (AS needs a finishing time); leftover inventory is
  liquidated at the end with a small cost.

* Quotes are snapped to the $0.10 tick grid (floor the bid, ceil the ask), so the
  "through" and "touch" rules actually differ. On a continuous price grid they
  cannot: a float quote is never exactly equal to a printed price.
* A hard inventory cap stops quoting the side that would worsen the position.
* Leftover inventory is unwound at the prevailing half-spread plus slippage,
  not at a token flat fee.
* Experiment 5 quotes around a flow-adjusted fair value (EWMA of signed trade
  volume, beta fitted out-of-sample) to test whether adverse selection falls.

v2 additions: (1) a constant-skew baseline so AS is compared against something smarter than a
fixed spread, (2) Sharpe + day-clustered bootstrap CIs on PnL, Sharpe and RMS-inventory
reduction, (3) matched-risk comparison on the fixed-spread frontier, (4) quote-latency sweep,
(5) volatility-regime split, (6) multi-asset support: sizes, skews and fill-curve distances
are in USD notional / bps of price and the tick size is inferred from the data.

Known simplifications: one unit per side per second, no queue model, no
partial fills, fees applied after the fact (maker fee, in bps).
"""

import os
import re
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
SYMBOLS = (sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT").split(",")

LOT_USD = 1000.0            # notional of ONE inventory unit; makes assets comparable
DAY_SECONDS = 86400
SESSION_SECONDS = 3600      # 24 one-hour sessions per day
SKEW_BPS = 0.2              # AS skew per unit of inventory at session start, in bps of price
G_SWEEP_BPS = [0.05, 0.1, 0.2, 0.4, 0.8]
NAIVE_MULTS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]   # multiples of the matched half-spread

MAX_INV = 25                # hard inventory cap (units); at the cap we quote one side only
INV_CAP_SWEEP = [10, 15, 25]
LIQ_TICKS = 2.0             # extra slippage (ticks) beyond the half-spread when unwinding
SIGMA_HORIZON = 60          # seconds; sigma from 60s returns avoids bid-ask-bounce inflation
IMB_HALFLIFE = 5.0          # seconds, EWMA halflife for the scale-free flow-imbalance signal
IMB_HORIZON = 1             # seconds ahead the signal is fitted to predict; matches quote rate
MARKOUT_SECONDS = 10
FEE_BPS_LEVELS = [-1.0, 0.0, 0.5, 1.0, 2.0]   # maker fee; negative = rebate
DELTAS_BPS = np.array([0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
LATENCY_SWEEP = [0, 1, 2, 5]   # seconds of quote staleness
SEED = 42
COLS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]


# ----------------------------------------------------------------------
# 1. Load one day of trades -> per-second arrays
# ----------------------------------------------------------------------
def load_day(path):
    date = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path)).group(1)
    day_start = int(pd.Timestamp(date, tz="UTC").timestamp())

    probe = pd.read_csv(path, header=None, nrows=1)
    has_header = not str(probe.iloc[0, 0]).strip().lstrip("-").isdigit()
    df = pd.read_csv(path, header=None, names=COLS, skiprows=1 if has_header else 0,
                     usecols=["price", "qty", "time", "is_buyer_maker"],
                     dtype={"is_buyer_maker": str})

    if len(df) > 1000 and df["time"].nunique() < 100:
        raise ValueError(f"{path}: timestamps look truncated (saved through Excel?). "
                         "Use the original unzipped CSV, never re-save it from Excel.")

    t = df["time"].to_numpy(float)
    if np.nanmedian(t) > 1e14:          # microseconds -> milliseconds
        t = t / 1000.0
    order = np.argsort(t, kind="stable")
    sec = np.floor(t[order] / 1000.0).astype(np.int64) - day_start
    price = df["price"].to_numpy(float)[order]
    qty = df["qty"].to_numpy(float)[order]
    is_sell = (df["is_buyer_maker"].str.strip().str.lower() == "true").to_numpy()[order]

    d = pd.DataFrame({"sec": sec, "price": price, "qty": qty, "sell": is_sell})
    d = d[(d.sec >= 0) & (d.sec < DAY_SECONDS)]
    idx = np.arange(DAY_SECONDS)
    last = d.groupby("sec")["price"].last().reindex(idx)
    min_sell = d.loc[d.sell].groupby("sec")["price"].min().reindex(idx)
    max_buy = d.loc[~d.sell].groupby("sec")["price"].max().reindex(idx)
    if last.isna().all():
        raise ValueError(f"{path}: no trades fall inside {date} (UTC)")

    # scale-free order-flow imbalance: EWMA(signed volume) / EWMA(total volume), in [-1, 1].
    # A raw-volume signal isn't comparable across a 790k-trade day and a 4.9M-trade day;
    # this ratio is, so a beta fitted on one day's scale is meaningful on another's.
    d["signed"] = np.where(d["sell"], -d["qty"], d["qty"])
    sv = d.groupby("sec")["signed"].sum().reindex(idx).fillna(0.0)
    tv = d.groupby("sec")["qty"].sum().reindex(idx).fillna(0.0)
    ewma_s = sv.ewm(halflife=IMB_HALFLIFE, adjust=False).mean()
    ewma_t = tv.ewm(halflife=IMB_HALFLIFE, adjust=False).mean()
    imb = (ewma_s / ewma_t.replace(0.0, np.nan)).fillna(0.0).clip(-1, 1).to_numpy()

    uniq = np.unique(price[:2_000_000])
    gaps = np.round(np.diff(uniq), 8)
    tick = float(gaps[gaps > 0].min()) if (gaps > 0).any() else 0.01   # inferred tick size
    px = float(np.median(price))
    return {"date": date, "n_trades": len(d),
            "mid": last.ffill().bfill().to_numpy(),
            "traded": last.notna().to_numpy(),
            "min_sell": min_sell.to_numpy(), "max_buy": max_buy.to_numpy(),
            "imb": imb, "tick": tick, "px": px}


# ----------------------------------------------------------------------
# 2. Calibrate sigma, kappa, A from one day
# ----------------------------------------------------------------------
def calibrate(day):
    ref = day["mid"][:-1]                       # mid known at the start of second s
    ms, mb = day["min_sell"][1:], day["max_buy"][1:]   # trades printed during second s
    deltas = DELTAS_BPS * 1e-4 * day["px"]      # $ distances scaled to this day's price level
    p = np.zeros(len(deltas))
    with np.errstate(invalid="ignore"):
        for i, dl in enumerate(deltas):
            p[i] = 0.5 * ((ms < ref - dl).mean() + (mb > ref + dl).mean())
    lam = -np.log1p(-np.clip(p, 0, 0.999))      # fills per second at distance delta

    # sigma from SIGMA_HORIZON-second returns, rescaled to 1s. Tick-by-tick diffs of the
    # last trade price are dominated by bid-ask bounce and overstate true volatility.
    h = SIGMA_HORIZON
    sigma = float(np.std(day["mid"][h:] - day["mid"][:-h]) / np.sqrt(h))

    ok = (p > 0.005) & (p < 0.9)
    out = {"date": day["date"], "sigma": sigma, "kappa": np.nan, "A": np.nan,
           "r2": np.nan, "n_fit": int(ok.sum()), "max_err": np.nan,
           "beta": np.nan, "beta_r2": np.nan, "z": None, "fut": None, "lam": lam, "ok": ok,
           "deltas": deltas, "px": day["px"], "tick": day["tick"]}
    if ok.sum() >= 3:
        x, y = deltas[ok], np.log(lam[ok])
        coef = np.polyfit(x, y, 1)
        ss_res = ((y - np.polyval(coef, x)) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        fit = np.exp(np.polyval(coef, x))
        out.update(kappa=-coef[0], A=float(np.exp(coef[1])), r2=1 - ss_res / ss_tot,
                   # R^2 in log space is ~1 for any smooth decay; this is the honest number
                   max_err=float(np.max(np.abs(fit / lam[ok] - 1))))

    # Signal (z) and forward price change (fut) this day contributes to the POOLED,
    # expanding-window beta fit done in main(). A single day's beta (printed below as a
    # diagnostic) is not trustworthy: at IMB_HORIZON-second sampling with overlapping
    # windows, one day does not pin down a coefficient this small.
    k = IMB_HORIZON
    z, fut = day["imb"][:-k], day["mid"][k:] - day["mid"][:-k]
    out["z"], out["fut"] = z, fut
    if np.std(z) > 0 and len(z) > 1000:
        beta = float(np.dot(z, fut) / np.dot(z, z))
        r2 = 1 - ((fut - beta * z) ** 2).sum() / ((fut - fut.mean()) ** 2).sum()
        out.update(beta=beta, beta_r2=r2)
    return out


def pooled_beta(cals, upto):
    """OLS-through-origin beta of fut ~ z, pooled over cals[0:upto] (days strictly before
    the trading day), with its t-stat. This is the beta actually used for trading."""
    zs = np.concatenate([c["z"] for c in cals[:upto] if c["z"] is not None])
    fs = np.concatenate([c["fut"] for c in cals[:upto] if c["fut"] is not None])
    n = len(zs)
    if n < 1000 or np.std(zs) == 0:
        return {"beta": 0.0, "t": np.nan, "r2": np.nan, "n": n}
    beta = float(np.dot(zs, fs) / np.dot(zs, zs))
    resid = fs - beta * zs
    se = np.sqrt((resid ** 2).sum() / (n - 1)) / np.sqrt((zs ** 2).sum())
    t_stat = beta / se if se > 0 else np.nan
    ss_tot = ((fs - fs.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return {"beta": beta, "t": float(t_stat), "r2": float(r2), "n": n}


# ----------------------------------------------------------------------
# 3. Quoting strategies (all in $ per BTC; one "unit" of inventory = LOT_USD of notional)
# ----------------------------------------------------------------------
def gamma_from_g(g, sigma, n=SESSION_SECONDS):
    """g = $ of skew per unit inventory at session start = gamma * sigma^2 * T."""
    return g / (sigma ** 2 * n)


def as_avg_half_spread(g, gamma, kappa):
    return 0.5 * (g / 2 + (2 / gamma) * np.log1p(gamma / kappa))


def make_as(g, gamma, kappa):
    def quote(m, q, tau):
        res = m - q * g * tau                                   # reservation price
        half = 0.5 * (g * tau + (2 / gamma) * np.log1p(gamma / kappa))
        return res - half, res + half
    return quote


def make_naive(half):
    return lambda m, q, tau: (m - half, m + half)


def make_skew(g, half):
    """Constant inventory skew, no time decay (infinite-horizon style, as in Gueant-Lehalle).
    The natural 'is AS actually better than just skewing?' baseline."""
    def quote(m, q, tau):
        res = m - q * g
        return res - half, res + half
    return quote


# ----------------------------------------------------------------------
# 4. Session builder and simulator (vectorised across sessions)
# ----------------------------------------------------------------------
def build_sessions(days, params):
    pre, post, ms, mb, fv, sig, meta = [], [], [], [], [], [], []
    for d, p in zip(days, params):
        # signal known at the START of each second, i.e. lagged by one second
        lag = np.concatenate([[0.0], d["imb"][:-1]])
        pbeta = p.get("pbeta", 0.0)
        adj = pbeta * lag                        # $ shift applied to the quoting mid
        for j in range(DAY_SECONDS // SESSION_SECONDS):
            a, b = j * SESSION_SECONDS, (j + 1) * SESSION_SECONDS
            if d["traded"][a:b].mean() < 0.5:        # skip sessions with missing data
                continue
            pre.append(d["mid"][np.maximum(np.arange(a - 1, b - 1), 0)])
            post.append(d["mid"][a:b])
            ms.append(d["min_sell"][a:b])
            mb.append(d["max_buy"][a:b])
            fv.append(adj[a:b])
            sig.append(lag[a:b])
            meta.append((d["date"], j, p["sigma"], p["kappa"], p["px"], p["tick"]))
    meta = pd.DataFrame(meta, columns=["date", "session", "sigma", "kappa", "px", "tick"])
    S = {"pre": np.array(pre).T.copy(), "post": np.array(post).T.copy(),
         "ms": np.array(ms).T.copy(), "mb": np.array(mb).T.copy(),
         "fv": np.array(fv).T.copy(), "sig": np.array(sig).T.copy(),     # shape (N, M)
         "tick": meta["tick"].to_numpy(), "lot": LOT_USD / meta["px"].to_numpy()}
    return S, meta


def simulate(S, quoter, rule="through", liq_cost=None, use_fv=False, max_inv=None, latency=0):
    """liq_cost: $ per BTC paid to unwind leftover inventory (array or scalar).
    use_fv:    quote around the flow-adjusted fair value instead of the raw mid.
    max_inv:   hard position cap (units); defaults to MAX_INV.
    latency:   seconds of staleness in the mid the quoter sees (0 = quotes use last second's mid)."""
    max_inv = MAX_INV if max_inv is None else max_inv
    N, M = S["pre"].shape
    tick = S["tick"]
    if liq_cost is None:
        liq_cost = np.zeros(M)
    liq_cost = np.broadcast_to(np.asarray(liq_cost, float), (M,))
    q = np.zeros(M); cash = np.zeros(M); notional = np.zeros(M)
    peak = np.zeros(M); dd = np.zeros(M)
    sum_q2 = np.zeros(M); max_q = np.zeros(M); fills = np.zeros(M)
    edge = np.zeros(M); markout = np.zeros(M); capped = np.zeros(M)

    for t in range(N):
        tau = 1.0 - t / N
        m = S["pre"][max(t - latency, 0)]
        bid, ask = quoter(m + S["fv"][t] if use_fv else m, q, tau)

        # Snap to the exchange tick grid, never more aggressively than the model asked.
        # Without this the two fill rules below are mathematically identical, because a
        # continuous float quote is (almost) never exactly equal to a printed price.
        bid = np.floor(bid / tick) * tick
        ask = np.ceil(ask / tick) * tick

        # A real desk has a position limit: at the cap, stop quoting the side that
        # would make the position worse.
        live_b, live_a = q < max_inv, q > -max_inv
        capped += ~(live_b & live_a)

        with np.errstate(invalid="ignore"):
            if rule == "through":
                hb, ha = (S["ms"][t] < bid) & live_b, (S["mb"][t] > ask) & live_a
            else:
                hb, ha = (S["ms"][t] <= bid) & live_b, (S["mb"][t] >= ask) & live_a
        hb, ha = np.nan_to_num(hb).astype(bool), np.nan_to_num(ha).astype(bool)

        q += hb
        q -= ha
        cash += np.where(hb, -bid, 0.0) + np.where(ha, ask, 0.0)
        notional += np.where(hb, bid, 0.0) + np.where(ha, ask, 0.0)
        fills += hb.astype(float) + ha   # bool+bool is a logical OR, which undercounted double fills

        mh = S["post"][min(t + MARKOUT_SECONDS, N - 1)]
        mid = S["pre"][t]                       # edge is always measured against the raw mid
        edge += np.where(hb, (mid - bid) / mid * 1e4, 0.0) + \
                np.where(ha, (ask - mid) / mid * 1e4, 0.0)
        markout += np.where(hb, (mh - bid) / mid * 1e4, 0.0) + \
                   np.where(ha, (ask - mh) / mid * 1e4, 0.0)

        pnl_t = cash + q * S["post"][t] - liq_cost * np.abs(q)
        peak = np.maximum(peak, pnl_t)
        dd = np.minimum(dd, pnl_t - peak)
        sum_q2 += q * q
        max_q = np.maximum(max_q, np.abs(q))

    final = cash + q * S["post"][-1] - liq_cost * np.abs(q)
    safe = np.where(fills > 0, fills, np.nan)
    return {"pnl": final * S["lot"], "dd": dd * S["lot"], "rms_inv": np.sqrt(sum_q2 / N),
            "max_inv": max_q, "fills": fills, "notional": notional * S["lot"],
            "edge_bps": edge / safe, "markout_bps": markout / safe,
            "capped_frac": capped / N}


def fill_signal_ceiling(S, quoter, rule="through", max_inv=None):
    """The ceiling check: regress realized markout (bps, side-adjusted so + is good for us)
    on the raw flow signal known at the moment of each fill. Uses the SAME signal a
    fair-value shift would use, so its R^2 upper-bounds what any beta on this signal can
    capture -- if this R^2 is tiny, no rescaling of it will help."""
    max_inv = MAX_INV if max_inv is None else max_inv
    N, M = S["pre"].shape
    tick = S["tick"]
    q = np.zeros(M)
    sig_parts, mk_parts = [], []
    for t in range(N):
        tau = 1.0 - t / N
        m = S["pre"][t]
        bid, ask = quoter(m, q, tau)
        bid = np.floor(bid / tick) * tick
        ask = np.ceil(ask / tick) * tick
        live_b, live_a = q < max_inv, q > -max_inv
        with np.errstate(invalid="ignore"):
            if rule == "through":
                hb, ha = (S["ms"][t] < bid) & live_b, (S["mb"][t] > ask) & live_a
            else:
                hb, ha = (S["ms"][t] <= bid) & live_b, (S["mb"][t] >= ask) & live_a
        hb, ha = np.nan_to_num(hb).astype(bool), np.nan_to_num(ha).astype(bool)
        q += hb; q -= ha

        mh = S["post"][min(t + MARKOUT_SECONDS, N - 1)]
        mid = S["pre"][t]
        if hb.any():
            i = np.where(hb)[0]
            mk_parts.append((mh[i] - bid[i]) / mid[i] * 1e4)
            sig_parts.append(S["sig"][t][i])                 # bought: + signal = bullish, good
        if ha.any():
            i = np.where(ha)[0]
            mk_parts.append((ask[i] - mh[i]) / mid[i] * 1e4)
            sig_parts.append(-S["sig"][t][i])                # sold: flip so + signal = good here too

    sig = np.concatenate(sig_parts) if sig_parts else np.array([])
    mk = np.concatenate(mk_parts) if mk_parts else np.array([])
    if len(sig) < 1000 or np.std(sig) == 0:
        return None
    beta = float(np.dot(sig, mk) / np.dot(sig, sig))
    resid = mk - beta * sig
    se = np.sqrt((resid ** 2).sum() / (len(sig) - 1)) / np.sqrt((sig ** 2).sum())
    ss_tot = ((mk - mk.mean()) ** 2).sum()
    return {"n": len(sig), "beta": beta, "t": float(beta / se) if se > 0 else np.nan,
            "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
            "mean_markout": float(mk.mean())}


# ----------------------------------------------------------------------
# 5. Statistics helpers
# ----------------------------------------------------------------------
def cluster_boot_std_ratio(a, b, day_idx, rng, n_boot=2000):
    """95% CI for std(a)/std(b), resampling whole days (sessions in a day are correlated)."""
    groups = [np.where(day_idx == d)[0] for d in np.unique(day_idx)]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        out.append(a[idx].std(ddof=1) / b[idx].std(ddof=1))
    return np.percentile(out, [2.5, 97.5])


def cluster_boot_ci(fn, a, b, day_idx, rng, n_boot=2000):
    """95% CI for fn(a[idx], b[idx]) resampling whole days."""
    groups = [np.where(day_idx == d)[0] for d in np.unique(day_idx)]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        out.append(fn(a[idx], b[idx]))
    return np.percentile(out, [2.5, 97.5])


def sharpe(x):
    return float(x.mean() / x.std(ddof=1))


def col_mean(x):
    return float(np.nanmean(x))


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def run(symbol):
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{symbol}-trades-*.csv")) +
                   glob.glob(os.path.join(DATA_DIR, f"{symbol}-trades-*.zip")))
    if not files:
        sys.exit(f"No files like {symbol}-trades-YYYY-MM-DD.csv found in {DATA_DIR}")

    print(f"Loading {len(files)} day(s)...")
    days = []
    for f in files:
        d = load_day(f)
        print(f"  {d['date']}: {d['n_trades']:,} trades, "
              f"{d['traded'].mean():.0%} of seconds have a trade")
        days.append(d)
    cals = [calibrate(d) for d in days]

    print("\nCalibration by day (fill rate vs distance from mid: lambda = A*exp(-kappa*delta))")
    print(f"sigma uses {SIGMA_HORIZON}s returns; 'maxerr' is the worst relative error of the "
          f"fit in LEVEL space (log-space R^2 flatters any smooth decay)")
    print(f"{'date':<12}{'sigma $/sqrt(s)':>16}{'kappa (1/$)':>13}{'A (/s)':>9}{'R^2':>7}"
          f"{'n':>4}{'maxerr':>9}{'daybeta':>9}{'dayR2':>8}")
    print("  'daybeta'/'dayR2' fit that single day only -- a diagnostic, NOT what trades. "
          f"At {IMB_HORIZON}s sampling one day cannot pin down a coefficient this small.")
    for c in cals:
        print(f"{c['date']:<12}{c['sigma']:>16.3f}{c['kappa']:>13.3f}{c['A']:>9.2f}"
              f"{c['r2']:>7.2f}{c['n_fit']:>4d}{c['max_err']:>9.2f}"
              f"{c['beta']:>9.2f}{c['beta_r2']:>8.3f}")

    if len(days) >= 2:
        pairs = [(days[i + 1], i) for i in range(len(days) - 1)
                 if np.isfinite(cals[i]["kappa"]) and cals[i]["kappa"] > 0]
        print("\nParameters from day d-1 are used to trade day d (out-of-sample).")
    else:
        pairs = [(days[0], 0)] if np.isfinite(cals[0]["kappa"]) else []
        print("\nWARNING: only one day supplied, so calibration and trading use the SAME day "
              "(in-sample). Add more days for a real test.")
    if not pairs:
        sys.exit("Calibration failed (fill-rate curve could not be fitted).")

    # Pooled, expanding-window beta: trading day i+1 uses every prior day's (signal, forward
    # return) pairs pooled together, not just day i's. This is the beta actually used to shift
    # the quoting mid. Reported alongside its t-stat so a coefficient with no real signal
    # (t << 2) is visible rather than silently traded on.
    print("\nPooled (expanding-window) flow-imbalance beta actually used for trading:")
    print(f"{'trades on':<12}{'pooled beta':>12}{'t-stat':>9}{'pooled R^2':>11}{'n obs':>10}")
    trade_days, cal_idx = zip(*pairs)
    params = []
    for d, i in pairs:
        upto = i + 1 if len(days) >= 2 else 1
        pb = pooled_beta(cals, upto)
        p = dict(cals[i]); p["pbeta"] = pb["beta"]
        params.append(p)
        print(f"{d['date']:<12}{pb['beta']:>12.3f}{pb['t']:>9.2f}{pb['r2']:>11.4f}{pb['n']:>10,d}")

    S, meta = build_sessions(trade_days, params)
    day_idx = pd.factorize(meta["date"])[0]
    n_days, n_sess = len(set(day_idx)), len(meta)
    print(f"Sessions: {n_sess} ({n_days} trading day(s) x up to 24 one-hour sessions)")

    sigma, kappa = meta["sigma"].to_numpy(), meta["kappa"].to_numpy()
    px = meta["px"].to_numpy()
    g0 = SKEW_BPS * 1e-4 * px                 # $ skew per unit at session start
    gamma = gamma_from_g(g0, sigma)
    half_match = as_avg_half_spread(g0, gamma, kappa)

    # Unwinding crosses the spread: at minimum the prevailing half-spread plus slippage.
    # The old flat $1/BTC was far too cheap and quietly subsidised the high-inventory book.
    liq = half_match + LIQ_TICKS * S["tick"]

    R_as = simulate(S, make_as(g0, gamma, kappa), liq_cost=liq)
    R_ls = simulate(S, make_skew(g0 / 2, half_match), liq_cost=liq)
    R_nv = simulate(S, make_naive(half_match), liq_cost=liq)

    # ---- Experiment 1: main comparison ---------------------------------
    print("\n" + "=" * 78)
    print("Experiment 1: AS vs constant-skew vs fixed spread (all equal average width)")
    print(f"Unit = ${LOT_USD:,.0f} notional | rule = trade-through | fees = 0 | "
          f"avg half-spread ${half_match.mean():.4f} | AS skew {SKEW_BPS} bps/unit at start")
    print("Const-skew gets the SAME average skew as AS (half the starting skew, no time decay),")
    print("so AS vs Const-skew isolates whether the finite-horizon term adds anything.")
    print("=" * 78)
    Rs = {"AS": R_as, "Const-skew": R_ls, "Fixed": R_nv}

    def metric(R, key):
        if key == "std":
            return R["pnl"].std(ddof=1)
        if key == "sharpe":
            return sharpe(R["pnl"])
        if key == "adv":
            return col_mean(R["edge_bps"]) - col_mean(R["markout_bps"])
        return col_mean(R[key])

    rows = [("Mean PnL / session ($)", "pnl"), ("PnL std across sessions ($)", "std"),
            ("Sharpe (mean/std, per session)", "sharpe"), ("Avg max drawdown ($)", "dd"),
            ("RMS inventory (units)", "rms_inv"), ("Max |inventory| (units)", "max_inv"),
            ("Fraction of time at cap", "capped_frac"), ("Fills / session", "fills"),
            ("Quoted edge per fill (bps)", "edge_bps"),
            (f"{MARKOUT_SECONDS}s markout per fill (bps)", "markout_bps"),
            ("Adverse selection (bps)", "adv")]
    print(f"{'Metric':<34}" + "".join(f"{k:>14}" for k in Rs))
    for label, key in rows:
        print(f"{label:<34}" + "".join(f"{metric(R, key):>14.3f}" for R in Rs.values()))

    daily = pd.DataFrame({"d": day_idx, "as": R_as["pnl"], "nv": R_nv["pnl"]}).groupby("d").sum()
    print("-" * 78)
    for name, R_b in [("fixed", R_nv), ("const-skew", R_ls)]:
        diff = R_as["pnl"] - R_b["pnl"]
        red = 1 - R_as["rms_inv"].mean() / R_b["rms_inv"].mean()
        print(f"AS vs {name}:  RMS-inventory reduction {100 * red:.1f}%")
        if np.allclose(diff, 0):
            print("  PnL identical across strategies (no fills?) - check data.")
            continue
        print(f"  mean PnL diff ${diff.mean():+.4f} (paired t p = {stats.ttest_rel(R_as['pnl'], R_b['pnl'])[1]:.4f}) | "
              f"median diff ${np.median(diff):+.4f} (Wilcoxon p = {stats.wilcoxon(R_as['pnl'], R_b['pnl'])[1]:.4f})")
        if n_days >= 3:
            lo, hi = cluster_boot_ci(lambda x, y: 1 - x.mean() / y.mean(), R_as["rms_inv"], R_b["rms_inv"], day_idx, rng)
            print(f"  RMS-inventory reduction 95% CI (day bootstrap): [{100 * lo:.1f}%, {100 * hi:.1f}%]")
            lo, hi = cluster_boot_ci(lambda x, y: x.mean() - y.mean(), R_as["pnl"], R_b["pnl"], day_idx, rng)
            print(f"  mean PnL diff 95% CI: [${lo:+.3f}, ${hi:+.3f}]")
            lo, hi = cluster_boot_ci(lambda x, y: sharpe(x) - sharpe(y), R_as["pnl"], R_b["pnl"], day_idx, rng)
            print(f"  Sharpe diff 95% CI:   [{lo:+.3f}, {hi:+.3f}]  (positive = AS better risk-adjusted)")
            lo, hi = cluster_boot_std_ratio(R_as["pnl"], R_b["pnl"], day_idx, rng)
            print(f"  PnL std ratio AS/{name}: {R_as['pnl'].std(ddof=1) / R_b['pnl'].std(ddof=1):.3f} (95% CI {lo:.3f} to {hi:.3f})")
    if len(daily) >= 5:
        print(f"Paired t-test on DAILY PnL, AS vs fixed ({len(daily)} days): "
              f"p = {stats.ttest_rel(daily['as'], daily['nv'])[1]:.4f}")

    # ---- Experiment 2: fill-rule sensitivity ---------------------------
    print("\n" + "=" * 68)
    print("Experiment 2: fill-rule sensitivity (mean PnL / session, $)")
    print("=" * 68)
    R_as_t = simulate(S, make_as(g0, gamma, kappa), rule="touch", liq_cost=liq)
    R_nv_t = simulate(S, make_naive(half_match), rule="touch", liq_cost=liq)
    print(f"{'rule':<26}{'AS':>10}{'Fixed':>10}")
    print(f"{'through (conservative)':<26}{R_as['pnl'].mean():>10.3f}{R_nv['pnl'].mean():>10.3f}")
    print(f"{'touch (optimistic)':<26}{R_as_t['pnl'].mean():>10.3f}{R_nv_t['pnl'].mean():>10.3f}")

    # ---- Experiment 3: fees ---------------------------------------------
    print("\n" + "=" * 68)
    print("Experiment 3: maker fee sensitivity (mean net PnL / session, $)")
    print("=" * 68)
    print(f"{'fee (bps)':<12}{'AS':>10}{'Fixed':>10}")
    fee_rows = []
    for fee in FEE_BPS_LEVELS:
        a_net = (R_as["pnl"] - fee * 1e-4 * R_as["notional"]).mean()
        n_net = (R_nv["pnl"] - fee * 1e-4 * R_nv["notional"]).mean()
        fee_rows.append((fee, a_net, n_net))
        print(f"{fee:<12.1f}{a_net:>10.3f}{n_net:>10.3f}")
    be_a = 1e4 * R_as["pnl"].sum() / R_as["notional"].sum()
    be_n = 1e4 * R_nv["pnl"].sum() / R_nv["notional"].sum()
    print(f"Break-even maker fee: AS {be_a:.3f} bps | Fixed {be_n:.3f} bps. Anything above this "
          "turns the strategy negative -- report results at a realistic fee tier, not just fee = 0.")

    # ---- Experiment 4: frontier -----------------------------------------
    print("\n" + "=" * 68)
    print("Experiment 4: risk-return frontier (mean PnL vs PnL std, $ per session)")
    print("=" * 68)
    as_front, nv_front = [], []
    for gb in G_SWEEP_BPS:
        g = gb * 1e-4 * px
        gm = gamma_from_g(g, sigma)
        R = simulate(S, make_as(g, gm, kappa), liq_cost=liq)
        as_front.append((gb, R["pnl"].mean(), R["pnl"].std(ddof=1)))
    for mlt in NAIVE_MULTS:
        R = simulate(S, make_naive(mlt * half_match), liq_cost=liq)
        nv_front.append((mlt, R["pnl"].mean(), R["pnl"].std(ddof=1)))
    print("AS    (skew bps/unit, mean, std):", [(g, round(float(m), 3), round(float(s), 3)) for g, m, s in as_front])
    print("Fixed (x matched half, mean, std):", [(k, round(float(m), 3), round(float(s), 3)) for k, m, s in nv_front])

    # Matched-risk view: tune the fixed book's width until it has AS's PnL std, then compare means.
    nv_sorted = sorted((s_, m_) for _, m_, s_ in nv_front)
    xs_, ys_ = [x for x, _ in nv_sorted], [y for _, y in nv_sorted]
    as_std, as_mean = R_as["pnl"].std(ddof=1), R_as["pnl"].mean()
    if xs_[0] <= as_std <= xs_[-1]:
        fixed_at = float(np.interp(as_std, xs_, ys_))
        print(f"Matched risk: at PnL std ${as_std:.3f}, AS earns ${as_mean:.3f}/session; a fixed "
              f"spread tuned to the same std earns ~${fixed_at:.3f}  (AS edge ${as_mean - fixed_at:+.3f})")
    else:
        print(f"Matched risk: AS PnL std ${as_std:.3f} is outside the fixed frontier "
              f"[${xs_[0]:.3f}, ${xs_[-1]:.3f}] -- no tested fixed width reaches AS's risk level, so "
              "compare on Sharpe (Experiment 1) instead.")
        fixed_at = np.nan

    # ---- Experiment 5: does a fair-value shift cut adverse selection? ----
    print("\n" + "=" * 68)
    print("Experiment 5: quoting around flow-adjusted fair value vs the raw mid")
    print("(pooled expanding-window beta; signal lagged one second)")
    print("=" * 68)
    ceil = fill_signal_ceiling(S, make_as(g0, gamma, kappa))
    if ceil is not None:
        print(f"Ceiling check: regressing realized {MARKOUT_SECONDS}s markout on the signal "
              f"known at fill time (n={ceil['n']:,} fills)")
        print(f"  slope {ceil['beta']:.3f} bps/unit-signal, t = {ceil['t']:.2f}, "
              f"R^2 = {ceil['r2']:.4f}, mean markout {ceil['mean_markout']:.3f} bps")
        print("  R^2 here is the MOST any fair-value shift built from this signal could "
              "capture, since it uses perfect knowledge of the fitted relationship. "
              "A small R^2 means the signal doesn't explain the adverse selection -- "
              "book data or a faster signal would be needed, not a better beta.")
    R_fv = simulate(S, make_as(g0, gamma, kappa), liq_cost=liq, use_fv=True)
    print(f"{'Metric':<32}{'AS (mid)':>14}{'AS (fair value)':>18}")
    for label, key in [("Mean PnL / session ($)", "pnl"), ("Fills / session", "fills"),
                       ("Quoted edge per fill (bps)", "edge_bps"),
                       (f"{MARKOUT_SECONDS}s markout per fill (bps)", "markout_bps")]:
        print(f"{label:<32}{col_mean(R_as[key]):>14.3f}{col_mean(R_fv[key]):>18.3f}")
    adv_a = col_mean(R_as["edge_bps"]) - col_mean(R_as["markout_bps"])
    adv_f = col_mean(R_fv["edge_bps"]) - col_mean(R_fv["markout_bps"])
    print(f"{'Adverse selection (bps)':<32}{adv_a:>14.3f}{adv_f:>18.3f}")
    print(f"Adverse selection reduced by {100 * (1 - adv_f / adv_a):.1f}% "
          f"of the raw-mid level." if adv_a else "")

    # ---- Experiment 6: how much of the variance edge survives a tighter cap? ----
    print("\n" + "=" * 68)
    print("Experiment 6: inventory-cap sweep (mean PnL, PnL std, $ per session)")
    print("=" * 68)
    print(f"{'cap (units)':<14}{'AS mean':>10}{'AS std':>9}{'Fixed mean':>12}{'Fixed std':>11}"
          f"{'std ratio':>11}")
    for cap in INV_CAP_SWEEP:
        Ra = simulate(S, make_as(g0, gamma, kappa), liq_cost=liq, max_inv=cap)
        Rn = simulate(S, make_naive(half_match), liq_cost=liq, max_inv=cap)
        sa, sn = Ra["pnl"].std(ddof=1), Rn["pnl"].std(ddof=1)
        print(f"{cap:<14}{Ra['pnl'].mean():>10.3f}{sa:>9.3f}{Rn['pnl'].mean():>12.3f}"
              f"{sn:>11.3f}{sa / sn:>11.3f}")
    print("A cap this tight binds constantly for both books; if the std ratio here is close "
          "to the uncapped one, AS's variance advantage is real skew-driven risk control, not "
          "just an artifact of the naive book having more room to run before Experiment 1's cap.")

    # ---- Experiment 7: quote latency ------------------------------------
    print("\n" + "=" * 68)
    print("Experiment 7: quote latency (quoter sees a mid that is L seconds stale)")
    print("=" * 68)
    print(f"{'latency (s)':<13}{'AS mean':>10}{'AS std':>9}{'Fixed mean':>12}{'Fixed std':>11}{'AS-Fixed':>10}")
    for L in LATENCY_SWEEP:
        Ra = simulate(S, make_as(g0, gamma, kappa), liq_cost=liq, latency=L)
        Rn = simulate(S, make_naive(half_match), liq_cost=liq, latency=L)
        print(f"{L:<13}{Ra['pnl'].mean():>10.3f}{Ra['pnl'].std(ddof=1):>9.3f}"
              f"{Rn['pnl'].mean():>12.3f}{Rn['pnl'].std(ddof=1):>11.3f}"
              f"{Ra['pnl'].mean() - Rn['pnl'].mean():>10.3f}")

    # ---- Experiment 8: volatility regimes -------------------------------
    print("\n" + "=" * 68)
    print("Experiment 8: results by volatility regime (terciles of calibrated sigma)")
    print("=" * 68)
    q1, q2 = np.quantile(sigma, [1 / 3, 2 / 3])
    print(f"{'regime':<10}{'n':>5}{'AS mean':>10}{'Fix mean':>10}{'AS std':>9}{'Fix std':>9}{'RMS red.':>10}")
    for name, mask in [("low vol", sigma <= q1), ("mid vol", (sigma > q1) & (sigma <= q2)), ("high vol", sigma > q2)]:
        if mask.sum() < 10:
            continue
        pa, pn = R_as["pnl"][mask], R_nv["pnl"][mask]
        red = 1 - R_as["rms_inv"][mask].mean() / R_nv["rms_inv"][mask].mean()
        print(f"{name:<10}{mask.sum():>5}{pa.mean():>10.3f}{pn.mean():>10.3f}"
              f"{pa.std(ddof=1):>9.3f}{pn.std(ddof=1):>9.3f}{100 * red:>9.1f}%")

    # ---- Save outputs ---------------------------------------------------
    out = meta.copy()
    out["as_pnl"], out["fixed_pnl"] = R_as["pnl"], R_nv["pnl"]
    out["as_fills"], out["fixed_fills"] = R_as["fills"], R_nv["fills"]
    out["skew_pnl"] = R_ls["pnl"]
    out.to_csv(f"{symbol}_session_results.csv", index=False)
    pd.DataFrame([{k: c[k] for k in ("date", "sigma", "kappa", "A", "r2")} for c in cals]
                 ).to_csv(f"{symbol}_calibration_by_day.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    c0 = next(c for c in cals if np.isfinite(c["kappa"]))
    ax[0, 0].semilogy(c0["deltas"], np.maximum(c0["lam"], 1e-6), "o", label="measured")
    ax[0, 0].semilogy(c0["deltas"], c0["A"] * np.exp(-c0["kappa"] * c0["deltas"]), "-", label="exponential fit")
    ax[0, 0].set_xlabel("Distance from mid ($)")
    ax[0, 0].set_ylabel("Fills per second")
    ax[0, 0].set_title(f"Fill rate vs distance ({c0['date']}, R2={c0['r2']:.2f})")
    ax[0, 0].legend()

    ax[0, 1].plot([s for _, _, s in as_front], [m for _, m, _ in as_front], "o-", label="AS (varying skew)")
    ax[0, 1].plot([s for _, _, s in nv_front], [m for _, m, _ in nv_front], "s-", label="Fixed (varying width)")
    ax[0, 1].set_xlabel("PnL std across sessions ($)")
    ax[0, 1].set_ylabel("Mean PnL ($)")
    ax[0, 1].set_title("Risk-return frontier (real tape)")
    ax[0, 1].scatter([R_ls["pnl"].std(ddof=1)], [R_ls["pnl"].mean()], marker="^", s=60, color="k",
                     label="Const-skew", zorder=5)
    ax[0, 1].legend()

    xs = np.arange(len(daily))
    ax[1, 0].bar(xs - 0.2, daily["as"], 0.4, label="AS")
    ax[1, 0].bar(xs + 0.2, daily["nv"], 0.4, label="Fixed")
    ax[1, 0].set_xticks(xs)
    ax[1, 0].set_xticklabels(sorted(set(meta["date"])), rotation=45, ha="right", fontsize=7)
    ax[1, 0].set_ylabel("Daily PnL ($)")
    ax[1, 0].set_title("Daily PnL (fees = 0)")
    ax[1, 0].legend()

    ax[1, 1].plot([f for f, _, _ in fee_rows], [a for _, a, _ in fee_rows], "o-", label="AS")
    ax[1, 1].plot([f for f, _, _ in fee_rows], [n for _, _, n in fee_rows], "s-", label="Fixed")
    ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[1, 1].set_xlabel("Maker fee (bps)")
    ax[1, 1].set_ylabel("Mean net PnL / session ($)")
    ax[1, 1].set_title("Fee sensitivity")
    ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{symbol}_results.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {symbol}_results.png, {symbol}_session_results.csv, {symbol}_calibration_by_day.csv")
    return {"symbol": symbol, "sessions": n_sess, "days": n_days,
            "rms_red_vs_fixed_%": 100 * (1 - R_as["rms_inv"].mean() / R_nv["rms_inv"].mean()),
            "rms_red_vs_skew_%": 100 * (1 - R_as["rms_inv"].mean() / R_ls["rms_inv"].mean()),
            "as_pnl": R_as["pnl"].mean(), "fixed_pnl": R_nv["pnl"].mean(), "skew_pnl": R_ls["pnl"].mean(),
            "as_sharpe": sharpe(R_as["pnl"]), "fixed_sharpe": sharpe(R_nv["pnl"]), "skew_sharpe": sharpe(R_ls["pnl"]),
            "as_edge_at_matched_risk": as_mean - fixed_at}


def main():
    rows = []
    for sym in SYMBOLS:
        print("\n" + "#" * 78 + f"\n# {sym}\n" + "#" * 78)
        try:
            rows.append(run(sym))
        except SystemExit as e:
            print(f"Skipping {sym}: {e}")
    if len(rows) > 1:
        summ = pd.DataFrame(rows)
        print("\n" + "=" * 78 + "\nCross-asset summary\n" + "=" * 78)
        print(summ.round(3).to_string(index=False))
        summ.to_csv("cross_asset_summary.csv", index=False)


if __name__ == "__main__":
    main()
