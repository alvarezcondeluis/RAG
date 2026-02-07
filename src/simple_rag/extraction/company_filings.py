from typing import Dict, List, Optional
from ..models.company import CompanyEntity, FilingMetadata, FinancialMetric, IncomeStatement, FinancialSegment, InsiderTransaction, Filing10K, ExecutiveCompensation
from datetime import date
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
# Concept Mapping: Maps XBRL concepts (with prefix) to IncomeStatement Pydantic properties
CONCEPT_TO_PROPERTY = {
    # --- REVENUE (Top Line) ---
    # For Tech/Retail
    "us-gaap_Revenues": "revenue",
    "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax": "revenue", # Apple, Nvidia standard
    "us-gaap_SalesRevenueNet": "revenue",
    "us-gaap_RevenueFromContractWithCustomer": "revenue",
    
    # FOR BANKS: ONLY map the "Net" number to Revenue (Total Net Revenue)
    "us-gaap_RevenuesNetOfInterestExpense": "revenue",  # WFC: $82B - The correct Top Line for banks
    
    # --- COST OF SALES (Direct Costs) ---
    "us-gaap_CostOfRevenue": "cost_of_sales",
    "us-gaap_CostOfGoodsAndServicesSold": "cost_of_sales",
    "us-gaap_CostOfGoodsSold": "cost_of_sales",
    # Bank Specific Cost (Provision for loan losses)
    "us-gaap_ProvisionForLoanLeaseAndOtherLosses": "cost_of_sales", # WFC: $4.3B
    
    # --- GROSS PROFIT ---
    "us-gaap_GrossProfit": "gross_profit",
    # FOR BANKS: Net Interest Income is their "Gross Profit"
    "us-gaap_InterestIncomeExpenseNet": "gross_profit", # WFC: $47B - Changed from net_income
    
    # --- OPERATING EXPENSES ---
    # Only map TOTAL operating expenses, not components (R&D, SG&A, etc.)
    "us-gaap_OperatingExpenses": "operating_expenses",
    # Bank Specific Operating Expenses
    "us-gaap_NoninterestExpense": "operating_expenses", # WFC: $54B - Total non-interest expenses
    
    # --- OPERATING INCOME ---
    "us-gaap_OperatingIncomeLoss": "operating_income",
    
    # --- OTHER INCOME/EXPENSE ---
    "us-gaap_NonoperatingIncomeExpense": "other_income_expense",

    # --- PRE-TAX INCOME ---
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "pretax_income",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments": "pretax_income",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxes": "pretax_income",
    
    # --- INCOME TAXES ---
    "us-gaap_IncomeTaxExpenseBenefit": "provision_for_income_taxes",
    "us-gaap_IncomeTaxesPaid": "provision_for_income_taxes",
    
    # --- NET INCOME (Bottom Line) ---
    "us-gaap_NetIncomeLoss": "net_income",
    "us-gaap_ProfitLoss": "net_income",
    "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic": "net_income",
    
    # --- EARNINGS PER SHARE (EPS) ---
    "us-gaap_EarningsPerShareBasic": "basic_earnings_per_share",
    "us-gaap_EarningsPerShareDiluted": "diluted_earnings_per_share",

    # --- SHARES OUTSTANDING ---
    "us-gaap_WeightedAverageNumberOfSharesOutstandingBasic": "basic_shares_outstanding",
    "us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding": "diluted_shares_outstanding"
}

# Priority Concepts: These are DEFINITIVE totals that should overwrite partial values
# This ensures "Total Net Revenue" beats "Interest Income" if both are mapped to revenue
PRIORITY_CONCEPTS = {
    "us-gaap_RevenuesNetOfInterestExpense",  # Bank Total Revenue (should overwrite partial interest income)
    "us-gaap_NetIncomeLoss",                 # Real Bottom Line
    "us-gaap_OperatingExpenses",             # Real Total Operating Expenses
    "us-gaap_NoninterestExpense",            # Bank Total Operating Expenses
    "us-gaap_Revenues",                      # Standard Total Revenue
    "us-gaap_GrossProfit",                   # Total Gross Profit
}


class CompanyExtractor:
    """Extracts company information from filing data."""
    
    def __init__(self):
        self.companies: Dict[str, Company] = {}
    
    def extract_insider_transactions_batch(self, filing_batch: List[tuple]) -> List[InsiderTransaction]:
        """
        Process a batch of insider filings in parallel.
        Args: List of (summary, filing_url) tuples
        Returns: List of all transactions from the batch
        """
        all_transactions = []
        
        for summary, filing_url in filing_batch:
            try:
                transactions = self.extract_insider_transactions(summary, filing_url)
                all_transactions.extend(transactions)
            except Exception as e:
                print(f"Error processing filing in batch: {e}")
                continue
        
        return all_transactions
    
    def extract_insider_transactions(self, summary, filing_url: str = None) -> List[InsiderTransaction]:
        """
        Procesa un resumen de Form 4 y extrae múltiples transacciones individuales.
        Returns a list of InsiderTransaction objects.
        """
        transactions = []
        
        # Datos comunes para todas las líneas de este reporte
        report_date = summary.reporting_date
        insider_name = summary.insider_name
        position = summary.position
        form_type = summary.form_type
        # Las acciones restantes suelen reportarse al final del formulario, 
        # pero técnicamente cambian línea por línea. Usaremos el valor final del reporte.
        final_remaining_shares = int(summary.remaining_shares) if summary.remaining_shares else 0

        # ITERAR sobre la lista de transacciones dentro del documento
        tx_list = summary.transactions
        
        for tx in tx_list:
            
            # FILTRO 1: Ignorar derivados (opciones/RSUs) si solo te interesan las acciones reales
            # Si quieres ver cuándo le dan opciones, quita este if.
            # Generalmente, 'non-derivative' es lo que afecta el precio de la acción hoy.
            if tx.security_type == 'derivative':
                continue

            # Mapeo de Códigos SEC a algo legible
            # Try different possible attribute names for the transaction code
            code = getattr(tx, 'code', None) or getattr(tx, 'transaction_code', None) or getattr(tx, 'trans_code', None)
            if not code:
                continue  # Skip if no transaction code found
            code = code.upper()
            tx_type = "UNKNOWN"
            
            if code == 'P': tx_type = "BUY"       # Compra en mercado abierto (¡Importante!)
            elif code == 'S': tx_type = "SELL"    # Venta en mercado abierto (¡Importante!)
            elif code == 'A': tx_type = "GRANT"   # Premio/Bono de acciones
            elif code == 'M': tx_type = "VESTING" # Conversión de RSU a Acción (Rutinario)
            elif code == 'F': tx_type = "TAX"     # Retención por impuestos (Rutinario)
            elif code == 'G': tx_type = "GIFT"    # Regalo/Donación

            # Extracción segura de valores numéricos
            try:
                shares = int(tx.shares) if tx.shares else 0
                price = float(tx.price_per_share) if tx.price_per_share else 0.0
                # Si el precio es 0 (ej. Grants), el valor es 0
                value = shares * price 
            except (ValueError, TypeError):
                shares = 0
                price = 0.0
                value = 0.0

            # Crear el objeto para ESTA línea específica
            transaction_record = InsiderTransaction(
                date=str(report_date),
                insider_name=str(insider_name),
                position=str(position),
                transaction_type=tx_type,
                shares=shares,
                price=price,
                value=value,
                remaining_shares=final_remaining_shares, # Nota: Esto es el total al final del form
                filing_url=filing_url,
                form=form_type
            )

            # Añadir a la lista de transacciones
            transactions.append(transaction_record)

            # (Opcional) Print para depurar y ver qué estás guardando
            if tx_type in ["BUY", "SELL"]:
                icon = "🟢" if tx_type == "BUY" else "🔴"
                print(f"   {icon} {tx_type}: {shares} acciones @ ${price:.2f}")
        
        return transactions

    def extract_executive_compensation(self, company: CompanyEntity, obj, filing_metadata: FilingMetadata, filing_url: str = None) -> None:
        """
        Extract executive compensation data and store it in the appropriate Filing10K object.
        """
        ex_compensation = ExecutiveCompensation(
            url=filing_url,
            form=obj.form,
            text=str(obj),
            ceo_name=obj.peo_name if hasattr(obj, 'peo_name') else None,
            ceo_compensation=obj.peo_total_comp,
            ceo_actually_paid=obj.peo_actually_paid_comp,
            shareholder_return=obj.total_shareholder_return
        )
        
        # Get or create Filing10K for this filing date
        filing_date = filing_metadata.filing_date
        if filing_date not in company.filings_10k:
            company.filings_10k[filing_date] = Filing10K(
                filing_metadata=filing_metadata
            )
        
        # Store executive compensation in the filing
        company.executive_compensation = ex_compensation
    
    def process_income_statement_dict(self, company: CompanyEntity, income_statement_dict: Dict, filing_metadata: FilingMetadata, income_statement: str = None) -> None:
        """
        Process income statement dictionary into IncomeStatement models.
        Creates one IncomeStatement per period (date) and populates it with line items.
        Creates or updates a Filing10K object to store the income statements.
        """
        periods = income_statement_dict["periods"]
        line_items = income_statement_dict["line_items"]
        
        # Get or create Filing10K for this filing date
        filing_date = filing_metadata.filing_date
        if filing_date not in company.filings_10k:
            company.filings_10k[filing_date] = Filing10K(
                filing_metadata=filing_metadata,
                income_statement_text=income_statement
            )
        
        filing_10k = company.filings_10k[filing_date]
        
        # Create an IncomeStatement for each period
        for period_date_str in periods:
            period_date = date.fromisoformat(period_date_str)
            
            # Initialize income statement for this period
            income_stmt = IncomeStatement(
                period_end_date=period_date,
                fiscal_year=period_date.year,
            )
            mapped_count = 0
            skipped_count = 0
            # Process each line item
            for line_item in line_items:
                concept = line_item["concept"]
                label = line_item["label"]
                
                print(concept + "Label: " + label)
                # Map concept to property name
                property_name = CONCEPT_TO_PROPERTY.get(concept)
                
                if not property_name:
                    # Skip unmapped concepts
                    skipped_count += 1
                    print(f"  ⊘ SKIPPED (unmapped): {concept} - '{label}'")
                    continue
                
                # Get value for this period
                value = line_item["values"].get(period_date_str)
                
                if value is None:
                    print(f"  ⊘ SKIPPED (no value): {concept} → {property_name}")
                    continue
                
                # Handle EPS separately (they're floats, not FinancialMetric)
                if property_name in ["basic_earnings_per_share", "diluted_earnings_per_share"]:
                    setattr(income_stmt, property_name, float(value))
                    mapped_count += 1
                    print(f"  ✓ MAPPED (EPS): {concept} → {property_name} = ${value:.2f}")
                    continue
                
                # Create FinancialMetric with segments
                # segments is now a dict: {axis_name: [segment_list]}
                segments = []
                segment_count = 0
                segments_dict = line_item.get("segments", {})
                
                # Iterate through each axis and its segments
                for axis_name, segment_list in segments_dict.items():
                    for segment_data in segment_list:
                        segment_value = segment_data["values"].get(period_date_str)
                        if segment_value is not None:
                            segment = FinancialSegment(
                                label=segment_data["label"],
                                amount=float(segment_value),
                                axis=axis_name
                            )
                            segments.append(segment)
                            segment_count += 1
                
                # Check if this property already has a value
                existing_metric = getattr(income_stmt, property_name, None)
                
                # Determine if we should set/overwrite the value
                # Allow set if: (1) field is empty, OR (2) this concept is a high-priority "Master Total"
                should_set_value = (existing_metric is None or existing_metric.value is None) or (concept in PRIORITY_CONCEPTS)
                
                if should_set_value:
                    # Create the FinancialMetric
                    metric = FinancialMetric(
                        value=float(value),
                        label=label,
                        segments=segments
                    )
                    
                    # Set the property on the income statement
                    setattr(income_stmt, property_name, metric)
                    mapped_count += 1
                    
                    priority_flag = " [PRIORITY]" if concept in PRIORITY_CONCEPTS else ""
                    segment_info = f" ({segment_count} segments)" if segment_count > 0 else ""
                    print(f"  ✓ MAPPED{priority_flag}: {concept} → {property_name} = ${value:,.2f}{segment_info}")
                else:
                    # Skip because a value already exists and this is not a priority concept
                    print(f"  ⊘ SKIPPED (already set): {concept} → {property_name} (existing: ${existing_metric.value:,.2f})")
            
            # Add to filing's income statements
            filing_10k.income_statements[period_date] = income_stmt
            
            print(f"\n  Summary for {period_date_str}:")
            print(f"    - Mapped: {mapped_count} line items")
            print(f"    - Skipped: {skipped_count} line items")
        
        print(f"\n{'='*80}")
        print(f"COMPLETED: {len(periods)} income statement(s) added to {company.name}")
        print(f"{'='*80}\n")
    
    def extract_income_statement_dict(self, df: pd.DataFrame) -> IncomeStatement:
        """Extract income statement from filing data."""
        
        date_cols = [col for col in df.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]
        # Sort descending (newest first) to match your desired output
        date_cols = sorted(date_cols, reverse=True)
        
        # 2. Container for the final lines
        line_items_map = {}
        
        # 3. Iterate through the dataframe
        # We group by 'concept' so Parent (dim=False) and Children (dim=True) are processed together
        # Note: We filter for rows that have a 'concept' (skipping abstract headers if necessary)
        if 'concept' not in df.columns:
            return {"error": "DataFrame missing 'concept' column"}
        
        # Grouping allows us to handle Parents and Segments regardless of row order
        grouped = df.groupby('concept', sort=False)
        
        final_items = []

        for concept, group in grouped:
            # A. Find the PARENT row (Where dimension is False or NaN)
            # Some rows might identify as the main concept
            parent_rows = group[ (group['dimension'] == False) | (group['dimension'].isna()) ]
            
            if parent_rows.empty:
                continue
                
            # Take the first valid parent row found
            parent = parent_rows.iloc[0]
            
            # Skip if it's an abstract header (no numbers)
            if parent.get('abstract') == True:
                print(f"  [DEBUG] Skipping abstract concept: {concept} - '{parent.get('label')}'")
                continue

            # Build Parent Object
            item = {
                "concept": str(concept),
                "label": str(parent['label']),
                "values": {},
                "segments": {}
            }
            
            # Extract Parent Values
            for date in date_cols:
                val = parent.get(date)
                if pd.notna(val):
                    item["values"][date] = float(val)
            
            # B. Find the SEGMENT rows (Where dimension is True)
            segment_rows = group[ group['dimension'] == True ]
            
            for _, seg_row in segment_rows.iterrows():
                segment_obj = {
                    "label": str(seg_row['label']),
                    "values": {}
                } 

                # Extract Segment Values
                has_data = False
                for date in date_cols:
                    val = seg_row.get(date)
                    if pd.notna(val):
                        segment_obj["values"][date] = float(val)
                        has_data = True
                
                # Only add segment if it has data
                if has_data:
                    # Determine the axis for this segment
                    # Priority: dimension_axis > parse from dimension_label > default
                    axis_name = seg_row.get('dimension_axis')
                    
                    if pd.isna(axis_name) or axis_name is None:
                        # Parse from dimension_label (e.g., "srt:ProductOrServiceAxis: iPhone")
                        dim_label = seg_row.get('dimension_label')
                        if pd.notna(dim_label) and dim_label:
                            # Extract the axis part before the colon
                            parts = str(dim_label).split(':')
                            if len(parts) >= 2:
                                # Take the last axis mentioned (e.g., "StatementBusinessSegmentsAxis" from "srt:ConsolidationItemsAxis: Operating segments, us-gaap:StatementBusinessSegmentsAxis: Americas")
                                axis_candidates = [p.strip() for p in str(dim_label).split(',')]
                                for candidate in reversed(axis_candidates):
                                    if ':' in candidate:
                                        axis_name = candidate.split(':')[0].strip()
                                        break
                            else:
                                axis_name = parts[0].strip()
                        else:
                            axis_name = "Other"
                    
                    axis_name = str(axis_name)
                    
                    # Ensure the list exists for this axis
                    if axis_name not in item["segments"]:
                        item["segments"][axis_name] = []
                        
                    # Append to the specific axis list
                    item["segments"][axis_name].append(segment_obj)
            
            # Add the item to final_items (after processing all segments)
            final_items.append(item)

        # 4. Construct Final Output
        return {
            "periods": date_cols,
            "line_items": final_items
        }


