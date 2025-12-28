import re
import pandas as pd

class XBRLUtils:
    """Static utilities for handling raw XBRL/HTML tasks."""
    
    @staticmethod
    def clean_text(tag) -> str:
        if not tag: return "N/A"
        return tag.text.strip().replace('\n', ' ')

    @staticmethod
    def stitch_block(start_tag, soup) -> str:
        """Reconstructs fragmented HTML blocks via 'continuedAt'."""
        if not start_tag: return ""
        
        full_html = str(start_tag)
        current_tag = start_tag
        
        # Limit 50 to prevent infinite loops
        for _ in range(50):
            next_id = current_tag.get("continuedat") or current_tag.get("continuedAt")
            if not next_id: break
            
            next_tag = soup.find(id=next_id)
            if next_tag:
                full_html += str(next_tag)
                current_tag = next_tag
            else:
                break
        return full_html


    @staticmethod
    def classify_table(df: pd.DataFrame) -> str:
        """Heuristic to determine what kind of table a DataFrame is."""
        
        try:
            if df.empty: return "Unknown"
        
            df_str = df.to_string().lower()
            headers = [str(col).lower() for col in df.columns]
            headers.append(str(df.iloc[0, 0]).lower())
            
           
            if "sector" in headers: return "Sector Allocation"
            if any(x in str(headers) for x in ["security", "holding", "state"]): return "Top 10 Holdings"
            if any(x in str(headers) for x in ["asset type", "asset class", "investment type"]): return "Portfolio Composition"
            if "geographic" in headers: return "Geographic Allocation"
            if "s&p credit ratingfootnote reference*" in headers or "moody's credit ratingfootnote reference*" in headers: return "Credit Rating"
            if "maturity" in headers: return "Maturity Allocation"
            if "issuer" in headers: return "Issuer Allocation"
            if "portfolio composition" in str(headers): return "Portfolio Composition"

            # Content Checks
            if "nvidia" in df_str or "inc." in df_str: return "Top 10 Holdings"
            if len(df) > 20: return "Performance Table"
            if "information technology" in df_str and len(df) < 5: return "Sector Allocation"
            if any(x in df_str for x in ["asset type", "asset class", "investment type"]): return "Portfolio Composition"

            if "maturity" in df_str: return "Maturity Allocation"
            if "country/geographic region" in df_str: return "Geographic Allocation"
            if "average annual" in df_str or "1 year" in df_str or "5 year" in df_str: return "Average Annual Returns"
            if "industry" in df_str: return "Industry Allocation"
            if "$10,000" in df_str or "Jun 20" in df_str: return "Performance Table"
            if "sector" in df_str or "other assets and liabilities—net" in df_str: return "Sector Allocation"
            if "credit rating*" in df_str: return "Credit Rating"
            if "issuer " in df_str: return "Issuer Allocation"
            
            
            print("Unknown Table: ", df)
            return "Unknown"
        except Exception as e:
            print("Failed to classify table: ", e)
            return "Unknown"

    