import numpy as np
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Retirement Portfolio Analysis')

    # Financial inputs
    parser.add_argument('--monthly-contribution', type=float, default=1000,
                        help='Monthly contribution amount (default: $1,000)')
    parser.add_argument('--initial-portfolio', type=float, default=1000000,
                        help='Initial portfolio value (default: $1,000,000)')
    parser.add_argument('--monthly-withdrawal', type=float, default=10000,
                        help='Monthly withdrawal in retirement (default: $10,000)')

    # Portfolio allocation
    parser.add_argument('--taxable-allocation', type=float, default=0.5,
                        help='Percentage of portfolio in taxable accounts (default: 0.5)')
    parser.add_argument('--traditional-allocation', type=float, default=0.4,
                        help='Percentage of portfolio in traditional IRA (default: 0.4)')
    parser.add_argument('--roth-allocation', type=float, default=0.1,
                        help='Percentage of portfolio in Roth IRA (default: 0.1)')

    # Asset allocation
    parser.add_argument('--stock-allocation', type=float, default=0.70,
                        help='Percentage of portfolio in stocks (default: 0.70)')
    parser.add_argument('--bond-allocation', type=float, default=0.25,
                        help='Percentage of portfolio in bonds (default: 0.25)')
    parser.add_argument('--cash-allocation', type=float, default=0.05,
                        help='Percentage of portfolio in cash (default: 0.05)')

    # Age parameters
    parser.add_argument('--current-age', type=int, default=40,
                        help='Current age (default: 40)')
    parser.add_argument('--retirement-age', type=int, default=65,
                        help='Retirement age (default: 65)')

    # Tax and inflation parameters
    parser.add_argument('--inflation-rate', type=float, default=0.03,
                        help='Annual inflation rate (default: 0.03)')
    parser.add_argument('--filing-status', type=str, default='married',
                        choices=['single', 'married'],
                        help='Tax filing status (default: married)')
    parser.add_argument('--state-tax-rate', type=float, default=0.05,
                        help='State tax rate (default: 0.05)')

    args = parser.parse_args()

    # Validate portfolio allocations sum to 1
    total_allocation = args.taxable_allocation + \
        args.traditional_allocation + args.roth_allocation
    if abs(total_allocation - 1.0) > 0.0001:  # Allow for small floating point differences
        parser.error("Portfolio allocations must sum to 1.0")

    # Validate asset allocations sum to 1
    total_asset_allocation = args.stock_allocation + \
        args.bond_allocation + args.cash_allocation
    if abs(total_asset_allocation - 1.0) > 0.0001:
        parser.error("Asset allocations must sum to 1.0")

    return args


class MarketAssumptions:
    def __init__(self):
        # Historical means and standard deviations
        self.stock_mean = 0.09
        self.bond_mean = 0.04
        self.stock_std = 0.15
        self.bond_std = 0.06
        # Correlation matrix between stocks and bonds
        self.correlation = -0.2  # Negative correlation for diversification benefit


@dataclass
class Portfolio:
    # Account types
    taxable: float
    traditional_ira: float
    roth_ira: float

    # Asset allocation (as percentages)
    stock_allocation: float = 0.70
    bond_allocation: float = 0.25
    cash_allocation: float = 0.05

    @property
    def total(self):
        return self.taxable + self.traditional_ira + self.roth_ira

    def get_allocation(self):
        return {
            'stocks': self.stock_allocation,
            'bonds': self.bond_allocation,
            'cash': self.cash_allocation
        }


class RMDCalculator:
    def __init__(self):
        # 2023 IRS Uniform Lifetime Table
        self.rmd_factors = {
            72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9,
            78: 22.0, 79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7,
            84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9,
            90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
            96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4
        }

    def calculate_rmd(self, age: int, traditional_ira_balance: float) -> float:
        if age < 72:  # RMD age is now 72
            return 0
        factor = self.rmd_factors.get(age, 6.4)  # Use 6.4 for ages over 100
        return traditional_ira_balance / factor


class PortfolioRebalancer:
    def __init__(self, target_allocation: dict, threshold: float = 0.05):
        self.target_allocation = target_allocation
        self.threshold = threshold

    def needs_rebalancing(self, current_allocation: dict) -> bool:
        return any(
            abs(current_allocation[asset] -
                self.target_allocation[asset]) > self.threshold
            for asset in self.target_allocation
        )

    def rebalance(self, portfolio_value: float) -> dict:
        return {
            asset: portfolio_value * allocation
            for asset, allocation in self.target_allocation.items()
        }


class SocialSecurityCalculator:
    def __init__(self):
        self.full_retirement_age = 67
        self.max_benefit_age = 70
        self.min_benefit_age = 62

    def calculate_benefit(self, retirement_age: int, earnings_history: List[float]) -> float:
        # Calculate AIME (Average Indexed Monthly Earnings)
        top_35_years = sorted(earnings_history, reverse=True)[:35]
        aime = sum(top_35_years) / 35 / 12

        # Calculate PIA (Primary Insurance Amount)
        if aime <= 1115:
            pia = aime * 0.90
        elif aime <= 6721:
            pia = 1003.50 + (aime - 1115) * 0.32
        else:
            pia = 2817.42 + (aime - 6721) * 0.15

        # Apply age-based adjustments
        if retirement_age < self.full_retirement_age:
            reduction = (self.full_retirement_age - retirement_age) * 0.067
            pia *= (1 - reduction)
        elif retirement_age > self.full_retirement_age:
            increase = (retirement_age - self.full_retirement_age) * 0.08
            pia *= (1 + increase)

        return pia * 12


class RetirementPlanner:
    def __init__(self):
        self.market = MarketAssumptions()

    def generate_returns(self, num_years, num_simulations):
        """Generate correlated returns for stocks and bonds"""
        # Create correlation matrix
        corr_matrix = np.array([[1, self.market.correlation],
                               [self.market.correlation, 1]])

        # Generate uncorrelated random numbers
        uncorrelated = np.random.normal(
            size=(2, num_simulations, num_years))

        # Create Cholesky decomposition of correlation matrix
        cholesky = np.linalg.cholesky(corr_matrix)

        # Generate correlated random numbers
        correlated = np.dot(cholesky, uncorrelated.reshape(
            2, -1)).reshape(2, num_simulations, num_years)

        # Transform to returns
        stock_returns = (self.market.stock_mean +
                         self.market.stock_std * correlated[0])
        bond_returns = (self.market.bond_mean +
                        self.market.bond_std * correlated[1])

        return stock_returns, bond_returns

    def simulate_portfolio(self, portfolio, num_years=30, num_simulations=1000):
        stock_returns, bond_returns = self.generate_returns(
            num_years, num_simulations)
        allocations = portfolio.get_allocation()

        portfolio_returns = (stock_returns * allocations['stocks'] +
                             bond_returns * allocations['bonds'])

        # Start with initial portfolio value
        initial_value = portfolio.total
        portfolio_values = np.zeros((num_simulations, num_years + 1))
        portfolio_values[:, 0] = initial_value

        # Simulate growth
        for year in range(num_years):
            portfolio_values[:, year + 1] = (portfolio_values[:, year] *
                                             (1 + portfolio_returns[:, year]))

        return portfolio_values


class WithdrawalStrategy:
    def __init__(self, initial_withdrawal: float, inflation_rate: float = 0.03):
        self.initial_withdrawal = initial_withdrawal
        self.inflation_rate = inflation_rate

    def constant_dollar(self, portfolio_value: float, year: int) -> float:
        """Traditional 4% rule with inflation adjustment"""
        return self.initial_withdrawal * (1 + self.inflation_rate) ** year

    def percent_of_portfolio(self, portfolio_value: float, year: int) -> float:
        """Withdraw a fixed percentage of current portfolio value"""
        return portfolio_value * 0.04

    def dynamic_spending(self, portfolio_value: float, year: int,
                         floor: float, ceiling: float) -> float:
        """Guyton-Klinger decision rules"""
        base_withdrawal = self.constant_dollar(portfolio_value, year)
        return min(max(base_withdrawal, floor), ceiling)


class RetirementSimulator:
    def __init__(self):
        self.market = MarketAssumptions()

    def simulate_with_withdrawals(self, portfolio, withdrawal_strategy,
                                  num_years=30, num_simulations=1000):
        portfolio_values = np.zeros((num_simulations, num_years + 1))
        withdrawal_amounts = np.zeros((num_simulations, num_years))
        success_rates = np.zeros(num_simulations)

        for sim in range(num_simulations):
            portfolio_value = portfolio.total
            for year in range(num_years):
                # Calculate withdrawal
                withdrawal = withdrawal_strategy.dynamic_spending(
                    portfolio_value, year,
                    floor=withdrawal_strategy.initial_withdrawal * 0.85,
                    ceiling=withdrawal_strategy.initial_withdrawal * 1.15
                )

                # Generate return for this year
                returns = self.generate_annual_returns(
                    portfolio.get_allocation())

                # Update portfolio value
                portfolio_value = (portfolio_value -
                                   withdrawal) * (1 + returns)

                # Store results
                portfolio_values[sim, year + 1] = portfolio_value
                withdrawal_amounts[sim, year] = withdrawal

            # Check if this simulation was successful (didn't run out of money)
            success_rates[sim] = portfolio_value > 0

        return {
            'portfolio_values': portfolio_values,
            'withdrawal_amounts': withdrawal_amounts,
            'success_rate': np.mean(success_rates)
        }

    def generate_annual_returns(self, allocation):
        """Generate single year returns based on allocation"""
        stock_return = np.random.normal(0.09, 0.15)  # Changed back to 9%
        bond_return = np.random.normal(0.04, 0.06)   # Changed back to 4%
        cash_return = 0.03  # Increased slightly for current environment

        return (stock_return * allocation['stocks'] +
                bond_return * allocation['bonds'] +
                cash_return * allocation['cash'])


class TaxStrategy:
    def __init__(self, tax_rates: dict):
        self.tax_rates = tax_rates

    def optimize_withdrawal(self, needed_amount: float, accounts: dict) -> dict:
        """Optimize withdrawals across account types for tax efficiency"""
        withdrawals = {}
        remaining = needed_amount

        # First use RMD if applicable
        if accounts.get('rmd_amount'):
            withdrawals['traditional'] = min(accounts['traditional'],
                                             accounts['rmd_amount'])
            remaining -= withdrawals['traditional']

        # Then use Roth up to standard deduction
        if remaining > 0 and accounts.get('roth'):
            tax_free_amount = self.tax_rates['standard_deduction']
            withdrawals['roth'] = min(accounts['roth'],
                                      remaining,
                                      tax_free_amount)
            remaining -= withdrawals['roth']

        # Then use taxable account gains
        if remaining > 0 and accounts.get('taxable'):
            withdrawals['taxable'] = min(accounts['taxable'],
                                         remaining)
            remaining -= withdrawals['taxable']

        return withdrawals


class SequenceRiskAnalyzer:
    def analyze_sequence_risk(self, portfolio_values: np.ndarray,
                              critical_years: int = 5) -> dict:
        """Analyze sequence of returns risk in early retirement years"""
        early_returns = portfolio_values[:, 1:critical_years+1] / \
            portfolio_values[:, 0:critical_years] - 1

        worst_sequences = np.argsort(np.mean(early_returns, axis=1))[:100]

        return {
            'worst_sequence_outcomes': portfolio_values[worst_sequences, -1],
            'worst_sequence_paths': portfolio_values[worst_sequences, :],
            'failure_rate': np.mean(portfolio_values[:, -1] <= 0)
        }


@dataclass
class TaxBracket:
    single_limits: List[float]
    married_limits: List[float]
    rates: List[float]


class ComprehensiveRetirementPlanner:
    def __init__(self,
                 current_age=40,
                 portfolio=Portfolio(500000, 400000, 100000),
                 monthly_contribution=1_000,
                 monthly_withdrawal=10_000,
                 inflation_rate=0.03,
                 filing_status='married',
                 state_tax_rate=0.05,
                 retirement_age=65):

        self.current_age = current_age
        self.portfolio = portfolio
        self.total_investments = portfolio.taxable + \
            portfolio.traditional_ira + portfolio.roth_ira
        self.monthly_contribution = monthly_contribution
        self.monthly_withdrawal = monthly_withdrawal
        self.inflation_rate = inflation_rate
        self.filing_status = filing_status
        self.state_tax_rate = state_tax_rate
        self.retirement_age = retirement_age

        self.tax_brackets = TaxBracket(
            single_limits=[11600, 47150, 100525,
                           191950, 243725, 609350, float('inf')],
            married_limits=[23200, 94300, 201050,
                            383900, 487450, 731200, float('inf')],
            rates=[0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
        )

    def calculate_social_security(self, retirement_age: int) -> float:
        base_benefit = 2000
        if retirement_age < 67:
            reduction = (67 - retirement_age) * 0.067
            base_benefit *= (1 - reduction)
        elif retirement_age > 67:
            increase = (retirement_age - 67) * 0.08
            base_benefit *= (1 + increase)
        return base_benefit * 12

    def analyze_success_rates(self, portfolio_simulations):
        """Analyze the probability of success"""
        final_values = portfolio_simulations[:, -1]
        success_rate = np.mean(final_values > 0)

        risk_analyzer = SequenceRiskAnalyzer()
        sequence_risk = risk_analyzer.analyze_sequence_risk(
            portfolio_simulations)

        print(f"\nSuccess Rate Analysis:")
        print(f"Probability of not running out of money: {success_rate:.1%}")
        print(
            f"Failure rate in worst sequence scenarios: {sequence_risk['failure_rate']:.1%}")

        return success_rate, sequence_risk

    def simulate_portfolio(self, stop_contribution_age, end_age=90, num_simulations=10000):
        years = end_age - self.current_age + 1
        portfolio_simulations = np.zeros((num_simulations, years))
        simulator = RetirementSimulator()

        # Set initial values
        portfolio_simulations[:, 0] = self.total_investments

        for sim in range(num_simulations):
            portfolio_value = self.total_investments

            for year in range(1, years):
                current_age = self.current_age + year

                # Apply return to existing portfolio
                annual_return = simulator.generate_annual_returns(
                    self.portfolio.get_allocation())
                portfolio_value = portfolio_value * (1 + annual_return)

                # Add contribution if applicable
                if current_age <= stop_contribution_age and current_age < self.retirement_age:
                    portfolio_value += self.monthly_contribution * 12

                # Apply withdrawals if in retirement
                if current_age >= self.retirement_age:
                    years_in_retirement = current_age - self.retirement_age
                    inflation_adjusted_withdrawal = (
                        self.monthly_withdrawal * 12 *
                        (1 + self.inflation_rate) ** years_in_retirement
                    )

                    ss_benefit = self.calculate_social_security(self.retirement_age) * \
                        (1 + self.inflation_rate) ** years_in_retirement

                    portfolio_value = max(
                        0, portfolio_value - (inflation_adjusted_withdrawal - ss_benefit))

                portfolio_simulations[sim, year] = portfolio_value

        return portfolio_simulations

    def analyze_contribution_impact(self):
        """Analyze portfolio outcomes with different contribution stop ages"""
        from scipy.stats import t

        np.random.seed(int(time.time()))
        stop_ages = range(self.current_age, self.retirement_age + 1)
        num_simulations = 10000
        results = []

        # Parameters for the t-distribution, black swan, and golden swan events
        df = 5  # Degrees of freedom for t-distribution
        black_swan_probability = 0.02  # 2% chance of black swan
        black_swan_impact = -0.4       # 40% drop in portfolio value
        golden_swan_probability = 0.02  # 2% chance of golden swan
        golden_swan_impact = 0.27       # 27% increase in portfolio value

        print("\n=== Retirement Portfolio Analysis ===")
        print(f"Initial Portfolio: ${self.total_investments:,.2f}")
        print(f"Current age: {self.current_age}")
        print(f"Retirement age: {self.retirement_age}")
        print(f"Monthly contribution: ${self.monthly_contribution:,.2f}")
        print(
            f"Monthly withdrawal in retirement: ${self.monthly_withdrawal:,.2f}")
        print(f"Number of simulations: {num_simulations:,}")
        print("\nAnalyzing contribution impact year by year...")
        print("\n{:<8} | {:>15} | {:>10} | {:>15} | {:>15} | {:>8} | {:>15} | {:>15} | {:>8}".format(
            "Age", "Portfolio", "Return %", "Return w/o Cont", "Return w/ Cont",
            "Impact %", "25th %tile", "75th %tile", "Range %"
        ))
        print("-" * 120)

        current_portfolio = self.total_investments

        for stop_age in stop_ages:
            simulations = np.zeros(num_simulations)
            returns_without_cont = np.zeros(num_simulations)
            returns_with_cont = np.zeros(num_simulations)

            for sim in range(num_simulations):
                # Determine if a black swan or golden swan event occurs
                rand_value = np.random.rand()
                if rand_value < black_swan_probability:
                    # Black swan event: apply a large negative return
                    annual_return = black_swan_impact
                elif rand_value < black_swan_probability + golden_swan_probability:
                    # Golden swan event: apply a large positive return
                    annual_return = golden_swan_impact
                else:
                    # Normal return using t-distribution
                    annual_return = t.rvs(
                        df,
                        loc=self.portfolio.stock_allocation * 0.09 +
                        self.portfolio.bond_allocation * 0.04 +
                        self.portfolio.cash_allocation * 0.03,
                        scale=0.15
                    )
                    # Cap the annual_return to realistic limits (-100% to +100%)
                    annual_return = max(min(annual_return, 1.0), -1.0)

                annual_contribution = self.monthly_contribution * \
                    12 if stop_age > self.current_age else 0

                # Portfolio WITHOUT contributions
                portfolio_without_cont = current_portfolio * \
                    (1 + annual_return)
                returns_without_cont[sim] = portfolio_without_cont - \
                    current_portfolio

                # Portfolio WITH contributions added BEFORE returns
                portfolio_with_contribution = current_portfolio + annual_contribution

                portfolio_with_cont = portfolio_with_contribution * \
                    (1 + annual_return)
                returns_with_cont[sim] = portfolio_with_cont - \
                    current_portfolio - annual_contribution

                simulations[sim] = portfolio_with_cont

            # Calculate median values and percentiles
            median_portfolio = np.median(simulations)
            median_return_without = np.median(returns_without_cont)
            median_return_with = np.median(returns_with_cont)
            percentile_25 = np.percentile(simulations, 25)
            percentile_75 = np.percentile(simulations, 75)
            range_pct = ((percentile_75 - percentile_25) /
                         median_portfolio) * 100

            # Calculate return percentage
            if stop_age == self.current_age:
                return_percentage = 0.0
            else:
                return_percentage = (median_portfolio - current_portfolio - annual_contribution) / (
                    current_portfolio + annual_contribution) * 100

            # Calculate the percentage difference in returns
            if median_return_without != 0:
                pct_difference = (
                    (median_return_with - median_return_without) / abs(median_return_without)) * 100
            else:
                pct_difference = 0

            if stop_age == self.current_age:
                print("{:<8} | ${:>14,.0f} | {:>9} | {:>15} | {:>15} | {:>7} | ${:>14,.0f} | ${:>14,.0f} | {:>7.1f}%".format(
                    stop_age, current_portfolio, "0.0", "Initial", "Initial", "N/A",
                    percentile_25, percentile_75, range_pct
                ))
            else:
                print("{:<8} | ${:>14,.0f} | {:>9.1f} | ${:>14,.0f} | ${:>14,.0f} | {:>7.1f}% | ${:>14,.0f} | ${:>14,.0f} | {:>7.1f}%".format(
                    stop_age, median_portfolio, return_percentage,
                    median_return_without, median_return_with, pct_difference,
                    percentile_25, percentile_75, range_pct
                ))

            if pct_difference < 2 and stop_age != self.current_age:
                print(
                    f"\n🎯 Recommendation: Consider stopping contributions at age {stop_age}")
                print(
                    f"   Reasoning: Additional contributions are only improving returns by {pct_difference:.1f}%")
                print(
                    f"   You could redirect ${self.monthly_contribution:,.2f} monthly to other goals")
                print(
                    f"   Portfolio range at this age: ${percentile_25:,.0f} to ${percentile_75:,.0f} (25-75th percentile)")
                break

            current_portfolio = median_portfolio

        return results


if __name__ == "__main__":
    args = parse_args()

    # Calculate portfolio allocations based on initial portfolio value
    taxable = args.initial_portfolio * 0.5  # 50% taxable
    traditional = args.initial_portfolio * 0.4  # 40% traditional IRA
    roth = args.initial_portfolio * 0.1  # 10% Roth IRA

    portfolio = Portfolio(
        taxable=taxable,
        traditional_ira=traditional,
        roth_ira=roth,
        stock_allocation=args.stock_allocation,
        bond_allocation=args.bond_allocation,
        cash_allocation=args.cash_allocation
    )

    planner = ComprehensiveRetirementPlanner(
        current_age=args.current_age,
        portfolio=portfolio,
        monthly_contribution=args.monthly_contribution,
        monthly_withdrawal=args.monthly_withdrawal,
        retirement_age=args.retirement_age,
        inflation_rate=args.inflation_rate,
        filing_status=args.filing_status,
        state_tax_rate=args.state_tax_rate
    )

    results = planner.analyze_contribution_impact()

    # After running the analysis, create Monte Carlo visualization
    plt.figure(figsize=(15, 10))

    # Run a full simulation through retirement
    portfolio_simulations = planner.simulate_portfolio(
        stop_contribution_age=planner.retirement_age)

    # Calculate percentiles for the simulation
    years = range(planner.current_age, 91)
    percentiles = np.percentile(portfolio_simulations, [
                                10, 25, 50, 75, 90], axis=0)

    # Create the Monte Carlo plot
    plt.fill_between(years, percentiles[0], percentiles[4], alpha=0.3,
                     color='gray', label='10-90th percentile')
    plt.fill_between(years, percentiles[1], percentiles[3], alpha=0.3,
                     color='blue', label='25-75th percentile')
    plt.plot(years, percentiles[2], 'r--', label='Median')

    # Add retirement age vertical line
    plt.axvline(x=planner.retirement_age, color='black', linestyle='--',
                label='Retirement Age')

    plt.title('Portfolio Value Projection (Monte Carlo Simulation)')
    plt.xlabel('Age')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)

    # Format y-axis in millions
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    plt.tight_layout()
    plt.show()
