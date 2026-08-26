# Optimal Market Making under Inventory Risk

Simulation and backtest of the Avellaneda-Stoikov (2008) optimal market-making
model against a naive fixed-spread baseline, on a Monte Carlo–simulated limit
order book.

## Motivation

Market makers earn the bid-ask spread but are exposed to inventory risk: if
they accumulate a large one-sided position, an adverse price move can wipe
out the spread income many times over. The Avellaneda-Stoikov (AS) framework
addresses this by deriving a **reservation price** that skews quotes away
from the mid-price as inventory builds, and an **optimal spread** that widens
as time-to-horizon and volatility increase. This project reproduces that
model from first principles and benchmarks it against a naive market maker
that quotes a symmetric fixed spread with no inventory awareness.

## Model

- **Mid-price**: simulated via Brownian motion, `dS = σ dW`
- **Reservation price**: `r(s, q, t) = s − q·γ·σ²·(T−t)`
- **Optimal spread**: `δ_a + δ_b = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/κ)`
- **Order arrivals**: Poisson process with intensity `λ(δ) = A·e^(−κδ)`,
  i.e. quotes closer to the mid-price fill more often

where `q` is current inventory, `γ` is the market maker's risk-aversion
coefficient, `σ` is volatility, `κ` controls order book liquidity/depth, and
`T−t` is time remaining in the trading session.

## Deliverables

1. **Simulation engine** (`market_maker_sim.py`) — GBM mid-price generator,
   Poisson fill-probability model, and two competing quoting strategies
   (Avellaneda-Stoikov vs. naive symmetric) run against the identical
   simulated tape for a fair comparison.
2. **Monte Carlo backtest** — 200 independent simulated trading sessions,
   each with 2,000 time steps, comparing terminal PnL, Sharpe ratio,
   inventory variance, and maximum drawdown across strategies.
3. **Results visualization** (`backtest_results.png`) — example-session PnL
   and inventory paths for both strategies, generated automatically on run.

## Results

Averaged over 200 simulated sessions:

| Metric              | Avellaneda-Stoikov | Naive Baseline |
|---------------------|--------------------|----------------|
| Mean terminal PnL    | 62.87              | 62.99          |
| Sharpe ratio         | 8.43               | 5.14           |
| Inventory variance   | 2.24               | 10.99          |
| Max drawdown         | -1.75              | -6.81          |

The AS strategy captures essentially the same spread income as the naive
baseline (mean PnL is statistically indistinguishable) but does so with
**64% higher risk-adjusted return (Sharpe)** and **80% lower inventory
variance**, confirming the core AS result: inventory-aware quoting doesn't
sacrifice edge, it sacrifices *variance in how that edge is realized*.

## Usage

```bash
pip install numpy matplotlib
python market_maker_sim.py
```

Adjust `GAMMA` (risk aversion), `SIGMA` (volatility), `K` (book liquidity),
`A` (base arrival rate), and `N_PATHS` (Monte Carlo sample size) at the top
of `market_maker_sim.py` to explore sensitivity.

## Possible Extensions

- Replace simulated GBM/Poisson flow with real tick-level order book data
  (e.g. crypto perpetuals or equities) for an out-of-sample test
- Add adverse selection: correlate order flow direction with short-term
  price momentum to test AS model robustness under informed trading
- Extend to multi-asset market making with correlated inventory risk

## References

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit
order book.* Quantitative Finance, 8(3), 217-224.
