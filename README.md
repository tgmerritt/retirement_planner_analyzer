# Retirement Portfolio Analyzer

A Python-based tool for analyzing retirement portfolio growth and contribution strategies. This tool uses Monte Carlo simulations to model different scenarios and provides recommendations on contribution strategies.

## Features

- Monte Carlo simulation with 10,000 iterations
- Portfolio analysis across different account types (Taxable, Traditional IRA, Roth IRA)
- Contribution impact analysis
- Tax consideration for different filing statuses
- Inflation adjustment
- Percentile-based risk analysis (25th-75th percentile range)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tgmerritt/retirement_planner_analyzer
cd retirement_planner_analyzer
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage with default parameters:
```bash
python analyze.py
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --monthly-contribution | 1000 | Monthly contribution amount in dollars |
| --initial-portfolio | 1000000 | Initial portfolio value in dollars |
| --monthly-withdrawal | 10000 | Monthly withdrawal amount in retirement |
| --taxable-allocation | 0.5 | Proportion of portfolio in taxable accounts (0-1) |
| --traditional-allocation | 0.4 | Proportion of portfolio in traditional IRA (0-1) |
| --roth-allocation | 0.1 | Proportion of portfolio in Roth IRA (0-1) |
| --current-age | 40 | Current age of the investor |
| --retirement-age | 65 | Planned retirement age |
| --inflation-rate | 0.03 | Annual inflation rate (as decimal) |
| --filing-status | married | Tax filing status ('single' or 'married') |
| --state-tax-rate | 0.05 | State tax rate (as decimal) |

### Example Usage Scenarios

1. Basic scenario with different contribution amount:
```bash
python analyze.py --monthly-contribution 1000
```

2. Larger initial portfolio with different allocation:
```bash
python analyze.py \
    --initial-portfolio 2000000 \
    --taxable-allocation 0.6 \
    --traditional-allocation 0.3 \
    --roth-allocation 0.1
```

3. Later retirement scenario:
```bash
python analyze.py \
    --current-age 50 \
    --retirement-age 70 \
    --monthly-contribution 6000
```

4. Single filer in high-tax state:
```bash
python analyze.py \
    --filing-status single \
    --state-tax-rate 0.08 \
    --monthly-contribution 3000
```

5. Comprehensive custom scenario:
```bash
python analyze.py \
    --monthly-contribution 6000 \
    --initial-portfolio 1500000 \
    --monthly-withdrawal 12000 \
    --taxable-allocation 0.4 \
    --traditional-allocation 0.4 \
    --roth-allocation 0.2 \
    --current-age 45 \
    --retirement-age 67 \
    --inflation-rate 0.035 \
    --filing-status single \
    --state-tax-rate 0.06
```

## Output Explanation

The tool provides:
- Portfolio value projections year by year
- Return percentages based on Monte Carlo simulations
- Impact of continued contributions
- 25th and 75th percentile ranges for risk assessment
- Recommendation on when to consider stopping contributions

Sample output:
```
=== Retirement Portfolio Analysis ===
Initial Portfolio: $1,000,000.00
Current age: 40
Retirement age: 65
Monthly contribution: $1,000.00
Monthly withdrawal in retirement: $10,000.00
...
```

## Notes

- Portfolio allocations (taxable, traditional, roth) must sum to 1.0
- Returns are simulated using historical market data patterns
- All monetary values should be input in dollars (no dollar signs needed)
- Rates (inflation, state tax) should be input as decimals (e.g., 0.03 for 3%)

### Explanation of Code Components:

#### Probability Check for Black Swan Events:

if np.random.rand() < black_swan_probability:
    # Black swan event occurs
np.random.rand() generates a random float between 0 and 1.
black_swan_probability is a predefined probability (e.g., 0.02 for a 2% chance).

If the random number is less than black_swan_probability, a black swan event is triggered.

#### Handling Black Swan Events:
    
`annual_return = black_swan_impact`

black_swan_impact is the predefined negative return representing a market crash (e.g., -0.4 for a 40% loss).

The annual return is set directly to this negative value when a black swan event occurs.

#### Calculating Normal Annual Returns:

```python
    else:
        # Normal return using t-distribution
        annual_return = t.rvs(
            df, 
            loc=...,
            scale=0.15
        )
```

Student's t-distribution (t.rvs):

Generates random numbers following a t-distribution, which has heavier tails than a normal distribution.

df is the degrees of freedom; lower values lead to heavier tails (more extreme values).

#### Parameters for t.rvs:

df: Degrees of freedom for the t-distribution (e.g., df = 5).
loc: The mean (expected value) of the distribution.
scale: The standard deviation (spread) of the distribution (e.g., 0.15).

#### Calculating the Expected Return (loc Parameter):

```python
loc = (
    self.portfolio.stock_allocation * 0.09 +
    self.portfolio.bond_allocation * 0.04 +
    self.portfolio.cash_allocation * 0.03
)
```

This calculates the weighted expected return based on the portfolio's asset allocation.

##### Components:

Stocks: Expected return of 9% (0.09).
Bonds: Expected return of 4% (0.04).
Cash: Expected return of 3% (0.03).

#### Weighting:

Each asset class's expected return is multiplied by its allocation percentage in the portfolio.

Example:

If the portfolio has 70% stocks, 25% bonds, and 5% cash:

```python
loc = (0.70 * 0.09) + (0.25 * 0.04) + (0.05 * 0.03)
    = 0.063 + 0.01 + 0.0015
    = 0.0745 (or 7.45% expected return)
```

#### Applying the Scale (Standard Deviation):

The scale parameter is set to 0.15, representing a 15% standard deviation, which introduces variability around the expected return (loc).

#### Capping the Annual Return:

`annual_return = max(min(annual_return, 1.0), -1.0)`

Ensures the annual return is within realistic bounds:

Minimum return: -100% (portfolio loses all value).

Maximum return: +100% (portfolio value doubles).

Prevents extreme outliers that could result from the t-distribution from skewing the simulation results unrealistically.

#### Putting It All Together:

Normal Market Conditions:

The annual return is drawn from a Student's t-distribution centered around the expected return calculated from the portfolio's allocations.

Variability is introduced through the scale (standard deviation) and the heavier tails of the t-distribution (df parameter).

This models typical market fluctuations, including both positive and negative returns.

#### Black Swan Events:

A small probability (black_swan_probability) introduces rare but significant negative returns to simulate market crashes or extreme downturns.

When a black swan event occurs, the annual return is set to a predefined negative impact (black_swan_impact).

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

Apache