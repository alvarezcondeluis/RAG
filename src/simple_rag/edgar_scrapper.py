import pandas as pd
from io import StringIO
from edgar import Company, set_identity
from pathlib import Path

# 1. Identity (Required)
set_identity("luis.alvarez.conde@alumnos.upm.es")

# 2. Get the Filing
# VOO is part of "Vanguard Index Funds" (CIK 0000036405)
fund = Company("VOO")
ncsr_filings = fund.get_filings(form="N-CSR")
latest_ncsr = ncsr_filings.latest()

print(f"Processing N-CSR filed on: {latest_ncsr.filing_date}")

# Create output directory
output_dir = Path("edgar_output")
output_dir.mkdir(exist_ok=True)

# 3. The "Object" Extraction Method

try:
    # Pandas reads the HTML and finds every <table> tag
    # This returns a LIST of DataFrame Objects
    tables = pd.read_html(StringIO(latest_ncsr.html()))

    print(f"Found {len(tables)} tables in the report.")

    # ---------------------------------------------------------
    # 4. FINDING THE USEFUL TABLES (Logic for your Graph)
    # ---------------------------------------------------------
    
    # A. Find "Sector Allocation" (The Portfolio Composition)
    # We loop through tables looking for keywords like "Communication Services"
    for i, df in enumerate(tables):
        # Convert whole dataframe to string to search keywords easily
        table_str = df.to_string()
        
        if "Communication Services" in table_str and "%" in table_str:
            print(f"\n--- FOUND SECTOR TABLE (Table index {i}) ---")
            # Clean up the dataframe (Column 0 is Sector, Column 1 is %)
            cleaned_df = df.dropna().reset_index(drop=True)
            print(cleaned_df)
            
            # Save sector table
            sector_txt = output_dir / f"VOO_sector_allocation_{latest_ncsr.filing_date}.txt"
            with open(sector_txt, "w", encoding="utf-8") as f:
                f.write("SECTOR ALLOCATION\n")
                f.write("=" * 80 + "\n\n")
                f.write(cleaned_df.to_string(index=False))
            print(f"Saved sector table to: {sector_txt}")
            
            break

    # B. Find "Performance" (Average Annual Total Returns)
    for i, df in enumerate(tables):
        if "Average Annual Total Returns" in str(df.columns) or "1 Year" in df.to_string():
            print(f"\n--- FOUND PERFORMANCE TABLE (Table index {i}) ---")
            print(df)
            
            # Save performance table
            perf_txt = output_dir / f"VOO_performance_{latest_ncsr.filing_date}.txt"
            
            with open(perf_txt, "w", encoding="utf-8") as f:
                f.write("PERFORMANCE - AVERAGE ANNUAL TOTAL RETURNS\n")
                f.write("=" * 80 + "\n\n")
                f.write(df.to_string(index=False))
            print(f"Saved performance table to: {perf_txt}")
            
            break

    # C. Find "Expense Ratios" (Financial Highlights)
    for i, df in enumerate(tables):
        if "Ratio of Total Expenses" in df.to_string():
            print(f"\n--- FOUND FINANCIAL HIGHLIGHTS (Table index {i}) ---")
            # This table usually has columns for 2024, 2023, 2022...
            print(df)
            
            # Save financial highlights table
            fin_txt = output_dir / f"VOO_financial_highlights_{latest_ncsr.filing_date}.txt"
            with open(fin_txt, "w", encoding="utf-8") as f:
                f.write("FINANCIAL HIGHLIGHTS - EXPENSE RATIOS\n")
                f.write("=" * 80 + "\n\n")
                f.write(df.to_string(index=False))
            print(f"Saved financial highlights to: {fin_txt}")
            
            break

    print(f"\n{'='*80}")
    print("SUMMARY - All files saved to 'edgar_output' directory:")
    print(f"{'='*80}")
    print(f"1. Sector Allocation: VOO_sector_allocation_{latest_ncsr.filing_date}.txt")
    print(f"2. Performance: VOO_performance_{latest_ncsr.filing_date}.txt")
    print(f"3. Financial Highlights: VOO_financial_highlights_{latest_ncsr.filing_date}.txt")
    print(f"{'='*80}\n")

except Exception as e:
    print(f"Error parsing HTML tables: {e}")