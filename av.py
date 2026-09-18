"""
Avellaneda-Stoikov market making on REAL Binance BTCUSDT futures trades.

Input : daily trade files named like BTCUSDT-trades-2026-09-01.csv (or .zip)
        columns: id, price, qty, quote_qty, time, is_buyer_maker
Run   : python real_data_backtest.py path/to/folder

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

LOT_BTC = 0.01              # size of one unit of inventory (BTC)
DAY_SECONDS = 86400
SESSION_SECONDS = 3600      # 24 one-hour sessions per day
SKEW_G = 2.0                # $ of quote skew per unit of inventory at session start
G_SWEEP = [0.5, 1, 2, 4, 8]
NAIVE_MULTS = [0.5, 0.75, 1.0, 1.5, 2.0]   # multiples of the matched half-spread

TICK = 0.10                 # BTCUSDT futures price tick ($). Quotes are snapped to it,
                            # otherwise "touch" and "through" can never differ.
MAX_INV = 25                # hard inventory cap (units); at the cap we quote one side only
INV_CAP_SWEEP = [10, 15, 25]
LIQ_TICKS = 2.0             # extra slippage (ticks) beyond the half-spread when unwinding
SIGMA_HORIZON = 60          # seconds; sigma from 60s returns avoids bid-ask-bounce inflation
IMB_HALFLIFE = 5.0          # seconds, EWMA halflife for the scale-free flow-imbalance signal
IMB_HORIZON = 1             # seconds ahead the signal is fitted to predict; matches quote rate
MARKOUT_SECONDS = 10
FEE_BPS_LEVELS = [-1.0, 0.0, 0.5, 1.0, 2.0]   # maker fee; negative = rebate
DELTAS = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20], float)  # $ from mid
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

    return {"date": date, "n_trades": len(d),
            "mid": last.ffill().bfill().to_numpy(),
            "traded": last.notna().to_numpy(),
            "min_sell": min_sell.to_numpy(), "max_buy": max_buy.to_numpy(),
            "imb": imb}


# ----------------------------------------------------------------------
# 2. Calibrate sigma, kappa, A from one day
# ----------------------------------------------------------------------
def calibrate(day):
    ref = day["mid"][:-1]                       # mid known at the start of second s
    ms, mb = day["min_sell"][1:], day["max_buy"][1:]   # trades printed during second s
    p = np.zeros(len(DELTAS))
    with np.errstate(invalid="ignore"):
        for i, dl in enumerate(DELTAS):
            p[i] = 0.5 * ((ms < ref - dl).mean() + (mb > ref + dl).mean())
    lam = -np.log1p(-np.clip(p, 0, 0.999))      # fills per second at distance delta

    # sigma from SIGMA_HORIZON-second returns, rescaled to 1s. Tick-by-tick diffs of the
    # last trade price are dominated by bid-ask bounce and overstate true volatility.
    h = SIGMA_HORIZON
    sigma = float(np.std(day["mid"][h:] - day["mid"][:-h]) / np.sqrt(h))

    ok = (p > 0.005) & (p < 0.9)
    out = {"date": day["date"], "sigma": sigma, "kappa": np.nan, "A": np.nan,
           "r2": np.nan, "n_fit": int(ok.sum()), "max_err": np.nan,
           "beta": np.nan, "beta_r2": np.nan, "z": None, "fut": None, "lam": lam, "ok": ok}
    if ok.sum() >= 3:
        x, y = DELTAS[ok], np.log(lam[ok])
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
# 3. Quoting strategies (all in $ per BTC; one "unit" of inventory = LOT_BTC)
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
            meta.append((d["date"], j, p["sigma"], p["kappa"]))
    meta = pd.DataFrame(meta, columns=["date", "session", "sigma", "kappa"])
    S = {"pre": np.array(pre).T.copy(), "post": np.array(post).T.copy(),
         "ms": np.array(ms).T.copy(), "mb": np.array(mb).T.copy(),
         "fv": np.array(fv).T.copy(), "sig": np.array(sig).T.copy()}     # shape (N, M)
    return S, meta


def simulate(S, quoter, rule="through", liq_cost=None, use_fv=False, max_inv=None):
    """liq_cost: $ per BTC paid to unwind leftover inventory (array or scalar).
    use_fv:    quote around the flow-adjusted fair value instead of the raw mid.
    max_inv:   hard position cap (units); defaults to MAX_INV."""
    max_inv = MAX_INV if max_inv is None else max_inv
    N, M = S["pre"].shape
    if liq_cost is None:
        liq_cost = np.zeros(M)
    liq_cost = np.broadcast_to(np.asarray(liq_cost, float), (M,))
    q = np.zeros(M); cash = np.zeros(M); notional = np.zeros(M)
    peak = np.zeros(M); dd = np.zeros(M)
    sum_q2 = np.zeros(M); max_q = np.zeros(M); fills = np.zeros(M)
    edge = np.zeros(M); markout = np.zeros(M); capped = np.zeros(M)

    for t in range(N):
        tau = 1.0 - t / N
        m = S["pre"][t]
        bid, ask = quoter(m + S["fv"][t] if use_fv else m, q, tau)

        # Snap to the exchange tick grid, never more aggressively than the model asked.
        # Without this the two fill rules below are mathematically identical, because a
        # continuous float quote is (almost) never exactly equal to a printed price.
        bid = np.floor(bid / TICK) * TICK
        ask = np.ceil(ask / TICK) * TICK

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
        fills += hb + ha

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
    return {"pnl": final * LOT_BTC, "dd": dd * LOT_BTC, "rms_inv": np.sqrt(sum_q2 / N),
            "max_inv": max_q, "fills": fills, "notional": notional * LOT_BTC,
            "edge_bps": edge / safe, "markout_bps": markout / safe,
            "capped_frac": capped / N}


def fill_signal_ceiling(S, quoter, rule="through", max_inv=None):
    """The ceiling check: regress realized markout (bps, side-adjusted so + is good for us)
    on the raw flow signal known at the moment of each fill. Uses the SAME signal a
    fair-value shift would use, so its R^2 upper-bounds what any beta on this signal can
    capture -- if this R^2 is tiny, no rescaling of it will help."""
    max_inv = MAX_INV if max_inv is None else max_inv
    N, M = S["pre"].shape
    q = np.zeros(M)
    sig_parts, mk_parts = [], []
    for t in range(N):
        tau = 1.0 - t / N
        m = S["pre"][t]
        bid, ask = quoter(m, q, tau)
        bid = np.floor(bid / TICK) * TICK
        ask = np.ceil(ask / TICK) * TICK
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


def col_mean(x):
    return float(np.nanmean(x))


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "BTCUSDT-trades-*.csv")) +
                   glob.glob(os.path.join(DATA_DIR, "BTCUSDT-trades-*.zip")))
    if not files:
        sys.exit(f"No files like BTCUSDT-trades-YYYY-MM-DD.csv found in {DATA_DIR}")

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
    gamma = gamma_from_g(SKEW_G, sigma)
    half_match = as_avg_half_spread(SKEW_G, gamma, kappa)

    # Unwinding crosses the spread: at minimum the prevailing half-spread plus slippage.
    # The old flat $1/BTC was far too cheap and quietly subsidised the high-inventory book.
    liq = half_match + LIQ_TICKS * TICK

    R_as = simulate(S, make_as(SKEW_G, gamma, kappa), liq_cost=liq)
    R_nv = simulate(S, make_naive(half_match), liq_cost=liq)

    # ---- Experiment 1: main comparison ---------------------------------
    print("\n" + "=" * 68)
    print(f"Experiment 1: AS (skew ${SKEW_G}/unit) vs fixed spread of equal average width")
    print(f"Unit = {LOT_BTC} BTC | rule = trade-through | fees = 0 | avg half-spread ${half_match.mean():.2f}")
    print("=" * 68)
    print(f"{'Metric':<32}{'AS':>14}{'Fixed spread':>16}")
    rows = [("Mean PnL / session ($)", "pnl"), ("PnL std across sessions ($)", None),
            ("Avg max drawdown ($)", "dd"), ("RMS inventory (units)", "rms_inv"),
            ("Max |inventory| (units)", "max_inv"), ("Fills / session", "fills"),
            ("Quoted edge per fill (bps)", "edge_bps"),
            (f"{MARKOUT_SECONDS}s markout per fill (bps)", "markout_bps")]
    for label, key in rows:
        if key is None:
            a_v, n_v = R_as["pnl"].std(ddof=1), R_nv["pnl"].std(ddof=1)
        else:
            a_v, n_v = col_mean(R_as[key]), col_mean(R_nv[key])
        print(f"{label:<32}{a_v:>14.3f}{n_v:>16.3f}")
    print(f"{'Adverse selection (edge-markout)':<32}"
          f"{col_mean(R_as['edge_bps']) - col_mean(R_as['markout_bps']):>14.3f}"
          f"{col_mean(R_nv['edge_bps']) - col_mean(R_nv['markout_bps']):>16.3f}   bps")

    diff = R_as["pnl"] - R_nv["pnl"]
    print("-" * 68)
    if np.allclose(diff, 0):
        print("PnL identical across strategies (no fills?) - check data.")
    else:
        print(f"Paired t-test on session PnL: p = {stats.ttest_rel(R_as['pnl'], R_nv['pnl'])[1]:.4f} "
              f"(mean diff ${diff.mean():+.4f})")
        print(f"Wilcoxon signed-rank:         p = {stats.wilcoxon(R_as['pnl'], R_nv['pnl'])[1]:.4f} "
              f"(median diff ${np.median(diff):+.4f})")
        print("  t-test compares MEANS and is swamped by the fixed book's fat tails; Wilcoxon "
              "compares the TYPICAL session. Disagreement between them is the result, not noise.")
    daily = pd.DataFrame({"d": day_idx, "as": R_as["pnl"], "nv": R_nv["pnl"]}).groupby("d").sum()
    if len(daily) >= 5:
        print(f"Paired t-test on DAILY PnL ({len(daily)} days): "
              f"p = {stats.ttest_rel(daily['as'], daily['nv'])[1]:.4f}")
    if n_days >= 3:
        lo, hi = cluster_boot_std_ratio(R_as["pnl"], R_nv["pnl"], day_idx, rng)
        print(f"PnL std ratio AS/fixed: {R_as['pnl'].std(ddof=1) / R_nv['pnl'].std(ddof=1):.3f} "
              f"(95% CI {lo:.3f} to {hi:.3f}, bootstrap over days)")

    # ---- Experiment 2: fill-rule sensitivity ---------------------------
    print("\n" + "=" * 68)
    print("Experiment 2: fill-rule sensitivity (mean PnL / session, $)")
    print("=" * 68)
    R_as_t = simulate(S, make_as(SKEW_G, gamma, kappa), rule="touch", liq_cost=liq)
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

    # ---- Experiment 4: frontier -----------------------------------------
    print("\n" + "=" * 68)
    print("Experiment 4: risk-return frontier (mean PnL vs PnL std, $ per session)")
    print("=" * 68)
    as_front, nv_front = [], []
    for g in G_SWEEP:
        gm = gamma_from_g(g, sigma)
        R = simulate(S, make_as(g, gm, kappa), liq_cost=liq)
        as_front.append((g, R["pnl"].mean(), R["pnl"].std(ddof=1)))
    for mlt in NAIVE_MULTS:
        R = simulate(S, make_naive(mlt * half_match), liq_cost=liq)
        nv_front.append((mlt, R["pnl"].mean(), R["pnl"].std(ddof=1)))
    print("AS    (skew $/unit, mean, std):", [(g, round(float(m), 3), round(float(s), 3)) for g, m, s in as_front])
    print("Fixed (x matched half, mean, std):", [(k, round(float(m), 3), round(float(s), 3)) for k, m, s in nv_front])

    # ---- Experiment 5: does a fair-value shift cut adverse selection? ----
    print("\n" + "=" * 68)
    print("Experiment 5: quoting around flow-adjusted fair value vs the raw mid")
    print("(pooled expanding-window beta; signal lagged one second)")
    print("=" * 68)
    ceil = fill_signal_ceiling(S, make_as(SKEW_G, gamma, kappa))
    if ceil is not None:
        print(f"Ceiling check: regressing realized {MARKOUT_SECONDS}s markout on the signal "
              f"known at fill time (n={ceil['n']:,} fills)")
        print(f"  slope {ceil['beta']:.3f} bps/unit-signal, t = {ceil['t']:.2f}, "
              f"R^2 = {ceil['r2']:.4f}, mean markout {ceil['mean_markout']:.3f} bps")
        print("  R^2 here is the MOST any fair-value shift built from this signal could "
              "capture, since it uses perfect knowledge of the fitted relationship. "
              "A small R^2 means the signal doesn't explain the adverse selection -- "
              "book data or a faster signal would be needed, not a better beta.")
    R_fv = simulate(S, make_as(SKEW_G, gamma, kappa), liq_cost=liq, use_fv=True)
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
        Ra = simulate(S, make_as(SKEW_G, gamma, kappa), liq_cost=liq, max_inv=cap)
        Rn = simulate(S, make_naive(half_match), liq_cost=liq, max_inv=cap)
        sa, sn = Ra["pnl"].std(ddof=1), Rn["pnl"].std(ddof=1)
        print(f"{cap:<14}{Ra['pnl'].mean():>10.3f}{sa:>9.3f}{Rn['pnl'].mean():>12.3f}"
              f"{sn:>11.3f}{sa / sn:>11.3f}")
    print("A cap this tight binds constantly for both books; if the std ratio here is close "
          "to the uncapped one, AS's variance advantage is real skew-driven risk control, not "
          "just an artifact of the naive book having more room to run before Experiment 1's cap.")

    # ---- Save outputs ---------------------------------------------------
    out = meta.copy()
    out["as_pnl"], out["fixed_pnl"] = R_as["pnl"], R_nv["pnl"]
    out["as_fills"], out["fixed_fills"] = R_as["fills"], R_nv["fills"]
    out.to_csv("session_results.csv", index=False)
    pd.DataFrame([{k: c[k] for k in ("date", "sigma", "kappa", "A", "r2")} for c in cals]
                 ).to_csv("calibration_by_day.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    c0 = next(c for c in cals if np.isfinite(c["kappa"]))
    ax[0, 0].semilogy(DELTAS, np.maximum(c0["lam"], 1e-6), "o", label="measured")
    ax[0, 0].semilogy(DELTAS, c0["A"] * np.exp(-c0["kappa"] * DELTAS), "-", label="exponential fit")
    ax[0, 0].set_xlabel("Distance from mid ($)")
    ax[0, 0].set_ylabel("Fills per second")
    ax[0, 0].set_title(f"Fill rate vs distance ({c0['date']}, R2={c0['r2']:.2f})")
    ax[0, 0].legend()

    ax[0, 1].plot([s for _, _, s in as_front], [m for _, m, _ in as_front], "o-", label="AS (varying skew)")
    ax[0, 1].plot([s for _, _, s in nv_front], [m for _, m, _ in nv_front], "s-", label="Fixed (varying width)")
    ax[0, 1].set_xlabel("PnL std across sessions ($)")
    ax[0, 1].set_ylabel("Mean PnL ($)")
    ax[0, 1].set_title("Risk-return frontier (real tape)")
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
    plt.savefig("real_data_results.png", dpi=150)
    print("\nSaved real_data_results.png, session_results.csv, calibration_by_day.csv")


if __name__ == "__main__":
    main()
