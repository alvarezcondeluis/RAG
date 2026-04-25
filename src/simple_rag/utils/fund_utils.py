from typing import List
from IPython.display import display, Markdown
from ..models.fund import FundData


def print_fund_info(fund_list: List[FundData]) -> None:
    print(f"Showing information of {len(fund_list)} funds")
    # Assuming 'funds' is your list of FundData objects
    for fund in fund_list:
        # 1. HEADER: Use Markdown for a nice visual separation
        display(Markdown(f"### 🏦 {fund.name}"))

        # 2. BASIC INFO GRID
        print(f"🆔 Context ID:      {fund.context_id}")
        print(f"🎫 Ticker:          {fund.ticker}")
        print(f"🏷️ Share Class:     {fund.share_class}")
        print(f"📅 Report Date:     {fund.report_date}")
        print(f"🏛️ Sec Exchange:    {fund.security_exchange}")
        print(f"📋 Registrant:     {fund.registrant}")

        # 3. FINANCIALS
        print("\n--- 💰 Costs & Financials ---")
        # Using f-string alignment (<20) to make columns line up perfectly
        print(f"{'Net Assets':<20}: {fund.net_assets}")
        print(f"{'Expense Ratio':<20}: {fund.expense_ratio}")
        print(f"{'Turnover Rate':<20}: {fund.turnover_rate}")
        print(f"{'Costs per $10k':<20}: {fund.costs_per_10k}")
        print(f"{'Advisory Fees':<20}: {fund.advisory_fees}")
        print(f"{'Number of Holdings':<20}: {fund.n_holdings}")

        # 4. COMMENTARY (Truncated)
        # We strip newlines and limit it to 200 chars so it doesn't clutter the screen
        if fund.performance_commentary and fund.performance_commentary != "N/A":
            clean_commentary = fund.performance_commentary.replace('\n', ' ').strip()
            print(f"\n📝 Commentary: \"{clean_commentary[:250]}...\"")

        # 5. DATA TABLES
        # We check each dataframe to see if it exists before displaying
        if fund.performance_table is not None:
            display(Markdown("**📈 Performance History**"))
            display(fund.performance_table)

        if fund.industry_allocation is not None:
            display(Markdown("**📊 Industry Allocation**"))
            display(fund.industry_allocation)

        if fund.avg_annual_returns is not None:
            display(Markdown("**📊 Average Annual Returns**"))
            display(fund.avg_annual_returns)

        if fund.top_holdings is not None:
            display(Markdown("**🏆 Top Holdings**"))
            display(fund.top_holdings)

        if fund.sector_allocation is not None:
            display(Markdown("**🏗️ Sector Allocation**"))
            display(fund.sector_allocation)

        if fund.portfolio_composition is not None:
            display(Markdown("**🍰 Portfolio Composition**"))
            display(fund.portfolio_composition)

        if fund.geographic_allocation is not None:
            display(Markdown("**🌍 Geographic Allocation**"))
            display(fund.geographic_allocation)

        if fund.maturity_allocation is not None:
            display(Markdown("**📊 Maturity Allocation**"))
            display(fund.maturity_allocation)

        if fund.credit_rating is not None:
            display(Markdown("**📊 Credit Rating**"))
            display(fund.credit_rating)

        if fund.issuer_allocation is not None:
            display(Markdown("**📊 Issuer Allocation**"))
            display(fund.issuer_allocation)

        # 6. END OF FUND SEPARATOR
        print("\n" + "="*80 + "\n")