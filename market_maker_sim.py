"""
Optimal Market Making under Inventory Risk
Avellaneda-Stoikov (2008) simulation & backtest vs. naive symmetric quoting.

Author: Dev M. Bandhiya

Model reference:
    Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit
    order book." Quantitative Finance, 8(3), 217-224.

Overview
--------
1. Simulate a mid-price path via Geometric Brownian Motion (GBM).
2. Simulate limit order arrivals on the bid/ask side via a Poisson process
   whose intensity decays exponentially with distance from the mid-price
   (the standard AS order-flow assumption).
3. Run two market makers against the SAME simulated tape:
     (a) Avellaneda-Stoikov: quotes around an inventory-skewed
         "reservation price" with a spread derived from risk aversion,
         volatility, and time-to-horizon.
     (b) Naive baseline: symmetric fixed-width quotes around the mid-price,
         no inventory awareness.
4. Compare PnL, Sharpe ratio, inventory variance, and max drawdown.

Run:
    python market_maker_sim.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Simulation parameters
# ----------------------------------------------------------------------
np.random.seed(42)

S0 = 100.0          # initial mid-price
SIGMA = 2.0          # annualized-equivalent volatility (per sqrt(T) units)
T = 1.0              # trading horizon (normalized, e.g. one trading day)
N_STEPS = 2000        # time discretization
DT = T / N_STEPS

GAMMA = 0.1          # risk aversion coefficient
K = 1.5              # order book liquidity / depth parameter
A = 140.0            # base arrival intensity (orders per unit time at delta=0)

N_PATHS = 200        # number of independent simulated trading sessions


# ----------------------------------------------------------------------
# 2. Core simulation for a single trading session
# ----------------------------------------------------------------------
def simulate_mid_price():
    """GBM mid-price path."""
    dW = np.random.normal(0, np.sqrt(DT), N_STEPS)
    S = np.zeros(N_STEPS + 1)
    S[0] = S0
    for t in range(N_STEPS):
        S[t + 1] = S[t] + SIGMA * dW[t]   # arithmetic BM (AS paper convention)
    return S


def fill_probability(delta, A=A, K=K):
    """Poisson fill intensity as a function of quote distance from mid."""
    return A * np.exp(-K * delta)


def run_avellaneda_stoikov(mid_prices):
    """Inventory-aware optimal market maker."""
    n = len(mid_prices)
    cash = 0.0
    inventory = 0
    inventory_path = np.zeros(n)
    pnl_path = np.zeros(n)

    for t in range(n - 1):
        tau = T - t * DT  # time remaining
        s = mid_prices[t]

        # reservation price: skews quotes away from mid based on inventory
        r = s - inventory * GAMMA * SIGMA ** 2 * tau

        # optimal total spread
        spread = GAMMA * SIGMA ** 2 * tau + (2 / GAMMA) * np.log(1 + GAMMA / K)
        half_spread = spread / 2

        bid = r - half_spread
        ask = r + half_spread

        delta_b = s - bid
        delta_a = ask - s

        lambda_b = fill_probability(delta_b)
        lambda_a = fill_probability(delta_a)

        p_fill_b = 1 - np.exp(-lambda_b * DT)
        p_fill_a = 1 - np.exp(-lambda_a * DT)

        if np.random.rand() < p_fill_b:
            inventory += 1
            cash -= bid
        if np.random.rand() < p_fill_a:
            inventory -= 1
            cash += ask

        inventory_path[t] = inventory
        pnl_path[t] = cash + inventory * mid_prices[t]

    inventory_path[-1] = inventory
    pnl_path[-1] = cash + inventory * mid_prices[-1]
    return pnl_path, inventory_path


def run_naive_baseline(mid_prices, fixed_half_spread=1.0):
    """Symmetric fixed-spread quoting, no inventory skew."""
    n = len(mid_prices)
    cash = 0.0
    inventory = 0
    inventory_path = np.zeros(n)
    pnl_path = np.zeros(n)

    for t in range(n - 1):
        s = mid_prices[t]
        bid = s - fixed_half_spread
        ask = s + fixed_half_spread

        lambda_b = fill_probability(fixed_half_spread)
        lambda_a = fill_probability(fixed_half_spread)

        p_fill_b = 1 - np.exp(-lambda_b * DT)
        p_fill_a = 1 - np.exp(-lambda_a * DT)

        if np.random.rand() < p_fill_b:
            inventory += 1
            cash -= bid
        if np.random.rand() < p_fill_a:
            inventory -= 1
            cash += ask

        inventory_path[t] = inventory
        pnl_path[t] = cash + inventory * mid_prices[t]

    inventory_path[-1] = inventory
    pnl_path[-1] = cash + inventory * mid_prices[-1]
    return pnl_path, inventory_path


# ----------------------------------------------------------------------
# 3. Monte Carlo backtest across many simulated sessions
# ----------------------------------------------------------------------
def sharpe_ratio(pnl_path):
    rets = np.diff(pnl_path)
    if rets.std() == 0:
        return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(N_STEPS)


def max_drawdown(pnl_path):
    running_max = np.maximum.accumulate(pnl_path)
    drawdown = pnl_path - running_max
    return drawdown.min()


def run_backtest():
    as_final_pnl, naive_final_pnl = [], []
    as_sharpes, naive_sharpes = [], []
    as_inv_var, naive_inv_var = [], []
    as_mdd, naive_mdd = [], []

    example_as_pnl, example_naive_pnl = None, None
    example_as_inv, example_naive_inv = None, None

    for path_idx in range(N_PATHS):
        mids = simulate_mid_price()

        as_pnl, as_inv = run_avellaneda_stoikov(mids)
        naive_pnl, naive_inv = run_naive_baseline(mids)

        as_final_pnl.append(as_pnl[-1])
        naive_final_pnl.append(naive_pnl[-1])

        as_sharpes.append(sharpe_ratio(as_pnl))
        naive_sharpes.append(sharpe_ratio(naive_pnl))

        as_inv_var.append(np.var(as_inv))
        naive_inv_var.append(np.var(naive_inv))

        as_mdd.append(max_drawdown(as_pnl))
        naive_mdd.append(max_drawdown(naive_pnl))

        if path_idx == 0:
            example_as_pnl, example_naive_pnl = as_pnl, naive_pnl
            example_as_inv, example_naive_inv = as_inv, naive_inv

    results = {
        "AS_mean_pnl": np.mean(as_final_pnl),
        "Naive_mean_pnl": np.mean(naive_final_pnl),
        "AS_sharpe": np.mean(as_sharpes),
        "Naive_sharpe": np.mean(naive_sharpes),
        "AS_inv_var": np.mean(as_inv_var),
        "Naive_inv_var": np.mean(naive_inv_var),
        "AS_mdd": np.mean(as_mdd),
        "Naive_mdd": np.mean(naive_mdd),
    }
    return results, (example_as_pnl, example_naive_pnl, example_as_inv, example_naive_inv)


if __name__ == "__main__":
    results, examples = run_backtest()
    example_as_pnl, example_naive_pnl, example_as_inv, example_naive_inv = examples

    print("=" * 60)
    print(f"Monte Carlo backtest over {N_PATHS} simulated sessions")
    print("=" * 60)
    print(f"{'Metric':<25}{'Avellaneda-Stoikov':<22}{'Naive Baseline'}")
    print(f"{'Mean terminal PnL':<25}{results['AS_mean_pnl']:<22.3f}{results['Naive_mean_pnl']:.3f}")
    print(f"{'Sharpe ratio':<25}{results['AS_sharpe']:<22.3f}{results['Naive_sharpe']:.3f}")
    print(f"{'Inventory variance':<25}{results['AS_inv_var']:<22.3f}{results['Naive_inv_var']:.3f}")
    print(f"{'Max drawdown':<25}{results['AS_mdd']:<22.3f}{results['Naive_mdd']:.3f}")

    sharpe_improvement = (results['AS_sharpe'] - results['Naive_sharpe']) / abs(results['Naive_sharpe']) * 100
    inv_reduction = (1 - results['AS_inv_var'] / results['Naive_inv_var']) * 100
    print("-" * 60)
    print(f"Sharpe improvement vs. naive: {sharpe_improvement:.1f}%")
    print(f"Inventory variance reduction: {inv_reduction:.1f}%")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(example_as_pnl, label="Avellaneda-Stoikov", linewidth=1.5)
    axes[0].plot(example_naive_pnl, label="Naive symmetric", linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel("PnL")
    axes[0].set_title("Example session: PnL path")
    axes[0].legend()

    axes[1].plot(example_as_inv, label="Avellaneda-Stoikov", linewidth=1.5)
    axes[1].plot(example_naive_inv, label="Naive symmetric", linewidth=1.5, alpha=0.8)
    axes[1].set_ylabel("Inventory")
    axes[1].set_xlabel("Time step")
    axes[1].set_title("Example session: inventory path")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150)
    print("\nSaved plot to backtest_results.png")
