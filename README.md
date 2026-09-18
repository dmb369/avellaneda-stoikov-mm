# Optimal Market Making under Inventory Risk

Backtest of the Avellaneda-Stoikov (2008) optimal market-making model
against a naive fixed-spread baseline, on real BTCUSDT futures tick data.

## Motivation

Market makers earn the bid-ask spread but are exposed to inventory risk: if
they accumulate a large one-sided position, an adverse price move can wipe
out the spread income many times over. The Avellaneda-Stoikov (AS) framework
addresses this by deriving a **reservation price** that skews quotes away
from the mid-price as inventory builds, and an **optimal spread** that widens
as time-to-horizon and volatility increase. This project implements that
model from first principles and backtests it against a naive market maker
that quotes a symmetric fixed spread with no inventory awareness — on real
tape, not a simulation calibrated to the model's own assumptions.

## Model

- **Reservation price**: `r(s, q, t) = s − q·γ·σ²·(T−t)`
- **Optimal spread**: `δ_a + δ_b = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/κ)`
- **Order arrivals**: Poisson process with intensity `λ(δ) = A·e^(−κδ)`,
  i.e. quotes closer to the mid-price fill more often

where `q` is current inventory, `γ` is the market maker's risk-aversion
coefficient, `σ` is volatility, `κ` controls order book liquidity/depth, and
`T−t` is time remaining in the trading session.

## Data and methodology

- **Input**: 17 days of BTCUSDT futures trade prints (~50M trades total),
  Binance format (`price`, `qty`, `time`, `is_buyer_maker`).
- **Mid-price proxy**: last trade price each second (no L2 book needed).
- **Fills**: a resting bid fills if a sell-side trade prints strictly below
  it ("trade-through"); `touch` (≤) is reported as an optimistic sensitivity.
  Quotes are snapped to the $0.10 exchange tick, so the two rules can
  actually differ.
- **Out-of-sample calibration**: volatility (`σ`, from 60s returns),
  fill-intensity (`κ`, `A`, fit to the empirical fill-rate-vs-distance
  curve), and the order-flow-imbalance beta (see below) are all estimated on
  prior days only and traded forward — day *d−1*'s parameters trade day *d*.
- **Risk controls**: a hard inventory cap (quoting stops on the side that
  would worsen the position past it), and leftover inventory is unwound at
  the prevailing half-spread plus slippage, not a token flat fee.
- **Sessions**: 1-hour trading windows (384 total across 16 out-of-sample
  trading days), each independently calibrated.

## Results

Averaged/aggregated over 384 real, out-of-sample 1-hour sessions:

| Metric                          | Avellaneda-Stoikov | Fixed spread |
|----------------------------------|--------------------|--------------|
| Mean PnL / session ($)           | -21.13              | -18.03        |
| PnL std across sessions ($)      | 27.34               | 44.72         |
| RMS inventory (units)            | 1.95                | 9.06          |
| Max \|inventory\| (units)        | 6.70                | 16.97         |
| Fills / session                  | 303                 | 293           |
| Adverse selection (edge − markout) | 1.63 bps          | 1.69 bps      |

**AS cut RMS inventory 78% and PnL variance 39%** (std ratio 0.611, 95% CI
0.545–0.777, bootstrapped over trading days) **relative to a fixed spread of
the same average width, at a statistically significant cost of about
$3/session in mean PnL** (paired t-test on daily PnL, p = 0.017).

This is the real AS trade-off, not a free lunch: inventory-aware quoting
buys materially lower variance and drawdown by giving up a small amount of
raw spread income, because skewing quotes away from a building position
means occasionally missing fills a symmetric quoter would have taken.
Neither strategy is profitable before fees at this spread/fee level — both
turn positive only with a maker rebate (~-1 bp), which is realistic: real
market makers are typically rebate-subsidized, not spread-only.

### Robustness checks

- **Fill rule**: "touch" vs "trade-through" fills change the result by
  <0.2%. The conclusion doesn't depend on the conservative assumption.
- **Fee sensitivity**: both strategies are monotonically worse as fees rise
  and only clear zero PnL under a maker rebate. AS remains behind Fixed at
  every fee level tested, consistent with the mean-PnL cost above.
- **Inventory cap sweep**: AS's variance advantage shrinks as the position
  cap tightens (std ratio 0.611 at a 25-unit cap → 0.895 at a 10-unit cap).
  A meaningful share of AS's headline risk reduction comes from *avoiding*
  large positions that a hard cap would otherwise have to cut off anyway —
  the two forms of risk control are partial substitutes, not independent.

### Does an order-flow signal explain the adverse selection?

Both strategies lose ~1.6-1.7 bps per fill to adverse selection (quoted edge
minus realized 10-second markout). We tested whether a short-horizon
order-flow-imbalance signal — EWMA(signed trade volume)/EWMA(total volume),
a scale-free ratio in [-1, 1] — could recover it by shifting the quoting
mid, with the beta fit on an **expanding, out-of-sample pooled window** of
all prior days (t-stat up to 43.6 by the final day — the signal itself is
statistically real).

It didn't help. A **ceiling regression** — realized markout against the
signal known at fill time, which upper-bounds what *any* beta on this
signal could capture — came back at R² ≈ 0 (t = -0.97 on 118,689 fills).
The signal predicts short-horizon price direction with a real but tiny
coefficient; it does not predict which of *our* fills will be adversely
selected. Recovering this adverse selection would require faster
(sub-second) signals or order-book depth, not a better-tuned beta on trade
prints. Applying the fair-value shift anyway changed PnL by <0.1%, exactly
as the ceiling check predicted — a case where testing the ceiling first
would have saved the effort of building the overlay.

## Repository contents

- `av.py` — loader, calibration, quoting strategies, vectorized simulator,
  and all six experiments below. Run with `python av.py path/to/data/`.
- `real_data_results.png` — fill-rate calibration, risk-return frontier,
  daily PnL, and fee-sensitivity plots, generated automatically on run.
- `session_results.csv`, `calibration_by_day.csv` — per-session and
  per-day numeric output.

### Experiments run

1. AS vs. fixed spread of equal average width (headline comparison above)
2. Fill-rule sensitivity (touch vs. trade-through)
3. Maker fee sensitivity (-1 to 2 bps)
4. Risk-return frontier (skew and spread-width sweeps)
5. Flow-adjusted fair value vs. raw mid, plus the ceiling regression
6. Inventory-cap sweep (10 / 15 / 25 units)

## Usage

```bash
pip install numpy pandas scipy matplotlib
python av.py path/to/data/   # expects BTCUSDT-trades-YYYY-MM-DD.csv files
```

Key parameters (top of `av.py`): `SKEW_G` (AS skew, $/unit inventory),
`MAX_INV` (position cap), `TICK` (exchange tick size), `SIGMA_HORIZON`
(volatility estimation window), `IMB_HALFLIFE`/`IMB_HORIZON` (flow-imbalance
signal), `FEE_BPS_LEVELS`, `INV_CAP_SWEEP`.

## Known simplifications

One unit per side per second, no queue-position model, no partial fills,
fees applied after the fact, last-trade-price mid proxy (no L2 book).

## Possible extensions

- Replace the trade-tape mid/fill proxy with real L2 order-book data —
  the ceiling check above suggests this, not a better trade-flow signal,
  is what's needed to close the adverse-selection gap
- Extend to multi-asset market making with correlated inventory risk
- Test position-cap and skew jointly rather than as separate sweeps, since
  Experiment 6 shows they substitute for each other

## References

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit
order book.* Quantitative Finance, 8(3), 217-224.
