# Optimal Market Making under Inventory Risk

Implementation and empirical evaluation of the Avellaneda–Stoikov (2008) optimal market-making model against a symmetric fixed-spread baseline, using 5-minute OHLC data across 13 liquid assets.

## Motivation

Market makers earn the bid–ask spread by buying at bid prices and selling at ask prices, but accumulating a large one-sided position exposes them to adverse price movements. A market maker can therefore earn spread income while simultaneously taking on substantial inventory risk.

The Avellaneda–Stoikov (AS) framework addresses this trade-off by adjusting quotes according to current inventory, volatility, time remaining, and market liquidity. Its reservation price shifts away from the mid-price as inventory builds, encouraging trades that reduce the position. Its optimal spread balances the compensation earned per fill against the probability of execution.

This project implements the model in Python and evaluates whether inventory-aware quoting improves risk-adjusted market-making performance relative to a symmetric fixed-spread strategy. The comparison uses historical market data, rolling parameter calibration, and out-of-sample evaluation across multiple asset classes.

## Model

### Reservation price

The AS reservation price adjusts the prevailing mid-price to account for inventory risk:

\[
r(s,q,t)=s-q\gamma\sigma^2(T-t)
\]

where:

- \(s\): current market mid-price.
- \(q\): current signed inventory; positive values indicate a long position.
- \(\gamma\): inventory risk-aversion coefficient.
- \(\sigma\): estimated price volatility.
- \(T-t\): remaining trading horizon.

When inventory is positive, the reservation price moves below the mid-price, encouraging selling and discouraging additional purchases. When inventory is negative, it moves above the mid-price, encouraging buying to reduce the short position.

### Optimal spread

The model derives the total optimal bid–ask spread as:

\[
\delta_a+\delta_b
=
\gamma\sigma^2(T-t)
+
\frac{2}{\gamma}
\ln\left(1+\frac{\gamma}{\kappa}\right)
\]

The first term compensates for inventory risk over the remaining horizon. The second reflects the trade-off between spread capture and execution probability.

The bid and ask quotes are constructed around the reservation price. In the symmetric-spread implementation:

\[
\text{Bid}=r-\frac{\delta}{2}
\]

\[
\text{Ask}=r+\frac{\delta}{2}
\]

where \(\delta=\delta_a+\delta_b\) is the total spread.

### Order-arrival model

The original AS framework models execution intensity as decreasing exponentially with quote distance:

\[
\lambda(\delta)=A e^{-\kappa\delta}
\]

Here, \(A\) controls the baseline arrival intensity and \(\kappa\) determines how quickly execution intensity falls as quotes move farther from the market.

The implementation calibrates volatility and fill-intensity decay using a trailing five-day window, with parameters estimated from historical observations before the out-of-sample trading period.

## Data and methodology

### Market data

- **Source:** Yahoo Finance historical market data.
- **Frequency:** 5-minute OHLC bars.
- **Universe:** 13 liquid assets spanning stocks, ETFs, futures, FX, and cryptocurrency.
- **Evaluation:** 2,044 out-of-sample trading sessions.
- **Calibration:** Trailing five-day historical window.

The multi-asset design tests whether inventory-aware quoting delivers consistent risk-management benefits across different markets rather than relying on results from a single cryptocurrency.

### Strategy comparison

Two strategies are evaluated:

**1. Avellaneda–Stoikov**

Quotes are adjusted according to the model's reservation price and risk-adjusted spread. Inventory accumulation changes the center of subsequent quotes, encouraging the strategy to manage its position actively.

**2. Symmetric fixed-spread baseline**

Quotes remain symmetric around the market mid-price, without inventory-dependent skew. The baseline uses the same average spread width as the AS strategy to make the comparison more informative.

Matching average spread width helps isolate the effect of inventory-aware quoting rather than attributing differences simply to one strategy quoting wider or narrower prices.

### Rolling calibration and out-of-sample testing

Volatility and fill-intensity decay parameters are estimated on a trailing five-day window and used for subsequent out-of-sample trading. This design limits look-ahead bias by ensuring that calibration relies on historical data available before the evaluated period.

The strategies are compared using session-level profit and loss (PnL), inventory exposure, and PnL volatility. Confidence intervals are constructed using a bootstrap over trading days to account for variation across days.

### Inventory and risk measurement

Inventory risk is evaluated using root-mean-square (RMS) inventory:

\[
q_{\mathrm{RMS}}
=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}q_i^2}
\]

This metric measures the typical magnitude of the position throughout a session. Lower RMS inventory indicates less sustained exposure to price movements.

PnL volatility is used to assess the variability of trading outcomes. Comparing it alongside mean PnL helps distinguish strategies that earn more on average from those that deliver more stable outcomes.

## Results

The backtest evaluates 2,044 out-of-sample sessions across 13 assets.

| Metric | Result |
|---|---|
| Inventory reduction | 12–30% lower RMS inventory |
| Asset coverage | All 13 assets |
| Out-of-sample sessions | 2,044 |
| PnL volatility | Lower for AS on every asset |
| Statistical significance of inventory reduction | 95% day-bootstrap confidence intervals excluded zero for all 13 assets |
| Statistically significant PnL cost | None detected for any asset |

### Key findings

**1. Inventory-aware quoting consistently reduces inventory exposure.**

The AS strategy reduced RMS inventory by 12–30% relative to the symmetric fixed-spread baseline across all 13 assets. The 95% day-bootstrap confidence intervals excluded zero for every asset, supporting the robustness of the observed inventory reduction.

**2. PnL volatility fell across the entire asset universe.**

The AS strategy produced lower PnL volatility on every asset tested. This is consistent with the intended mechanism of the model: shifting quotes in response to inventory discourages further accumulation of an already large position and encourages trades that reduce exposure.

**3. The inventory reduction did not produce a statistically significant PnL penalty.**

No asset showed a statistically significant deterioration in PnL relative to the baseline. This suggests that the reduction in inventory risk was not accompanied by a detectable loss in trading profitability under the tested methodology.

These findings support the value of inventory-aware quoting as a risk-management mechanism. They do not, by themselves, establish that AS generates higher absolute profits or remains profitable after all realistic execution costs.

## Statistical methodology

The comparison uses day-level bootstrap confidence intervals to assess the reliability of the observed inventory reduction. Resampling by day helps account for dependence among sessions occurring within the same trading day.

The analysis distinguishes two questions:

- **Inventory effectiveness:** Is the reduction in RMS inventory statistically distinguishable from zero?
- **PnL trade-off:** Is there evidence of a statistically significant deterioration in PnL?

The inventory reduction was statistically significant across all 13 assets, while no statistically significant PnL cost was detected for any asset. A failure to detect a significant PnL cost does not prove that the strategies have identical expected PnL; that stronger conclusion would require a suitably powered equivalence or non-inferiority test.

## Implementation

The project is implemented in Python, with the model, historical data processing, rolling calibration, backtesting, and performance analysis organized around the following workflow:

1. Load and preprocess historical 5-minute OHLC data.
2. Estimate volatility and fill-intensity parameters using a trailing five-day window.
3. Calculate the reservation price and risk-adjusted spread.
4. Generate inventory-aware AS quotes and symmetric fixed-spread benchmark quotes.
5. Simulate executions and update inventory and cash positions.
6. Calculate session-level PnL, RMS inventory, and PnL volatility.
7. Aggregate results across assets and evaluate statistical significance using day-level bootstrap confidence intervals.

## Limitations

Historical OHLC data do not provide the full information available from trade prints and order-book messages. Consequently, this backtest is a simplified evaluation of market-making behavior rather than a complete reconstruction of a live limit order book.

Important limitations include:

- **Execution uncertainty:** OHLC bars do not reveal queue position, order-book depth, or the exact sequence of trades within each bar.
- **Fill modeling:** Actual execution probabilities cannot be recovered perfectly from bar data. Results depend on the assumptions used to infer fills.
- **Market impact:** The framework does not fully model how the strategy's own orders might affect prices or other participants' behavior.
- **Transaction costs:** Commissions, exchange fees, rebates, slippage, and funding costs must be incorporated consistently before interpreting simulated PnL as achievable live performance.
- **Model assumptions:** The AS framework uses a simplified model of price dynamics and order arrivals. Real markets may exhibit jumps, autocorrelation, regime changes, and asymmetric order flow.
- **Parameter stability:** A trailing calibration window reduces look-ahead bias but does not eliminate sensitivity to market regimes or parameter estimation error.
- **Asset comparability:** Different instruments have different tick sizes, liquidity, trading hours, and contract specifications. Comparisons require appropriate normalization and instrument-specific handling.

Accordingly, the results demonstrate improvements in inventory management and PnL variability under the tested historical-data framework. They should not be interpreted as proof of live profitability.

## Possible extensions

- Replace OHLC-based execution proxies with trade-level data and Level 2 order-book data to model queue position, depth, and fill probabilities more accurately.
- Incorporate realistic exchange fees, maker rebates, slippage, funding, and inventory liquidation costs.
- Test alternative volatility estimators and adaptive calibration windows across changing market regimes.
- Optimize inventory risk aversion, spread width, and position limits jointly rather than tuning each independently.
- Extend the framework to multi-asset market making with correlated inventory risk.
- Evaluate performance using additional risk-adjusted metrics, including maximum drawdown, tail losses, and return-to-risk ratios.
- Conduct sensitivity analyses across calibration windows, quote update frequencies, and execution assumptions.
- Test the strategy in paper trading or a controlled live environment before drawing conclusions about deployability.

## Reference

Avellaneda, M., & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.
