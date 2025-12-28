"""
Chart generation utilities for fund data visualization.
"""

import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import re


def validate_and_clean_allocation(df, column: str, sort_by_value: bool = True):
    """
    Validate and clean allocation DataFrame.
    Returns cleaned df and prints validation report.
    
    Args:
        df: DataFrame to validate and clean
        column: Name of the category column
        sort_by_value: If True, sort by value descending. If False, keep original order.
    """
    if df is None or df.empty:
        print("⚠️ Empty DataFrame")
        return None
    if len(df.columns) > 2:
        
        df = df.iloc[:, :2]


    # Check first two rows to find the header row
    is_header = False
    header_row_idx = None
    
    for row_idx in range(min(2, len(df))):
        row_val = str(df.iloc[row_idx, 1])
        try:
            # Try to convert value to float (after cleaning)
            test_val = row_val.replace('%', '').replace('(', '').replace(')', '').strip()
            float(test_val)
        except (ValueError, AttributeError):
            # This row is likely a header
            new_columns = df.iloc[row_idx]
            
            # Check for duplicate column names
            if len(set(new_columns)) != len(new_columns):
                print(f"⚠️ Row {row_idx} has duplicate column names, skipping")
                continue
            
            # Valid header found
            is_header = True
            header_row_idx = row_idx
            print(f"🔍 Detected header row at index {row_idx}")
            df.columns = new_columns
            # Drop all rows up to and including the header row
            df = df.iloc[row_idx + 1:].reset_index(drop=True)
            print(f"New columns: {list(df.columns)}")
            break
    
    
    df_work = df.copy().reset_index(drop=True)
    if not is_header:
        df_work.columns = [column, 'Percent of Total Investments(a)']
    
    
    col_category = df_work.columns[0]
    col_value = df_work.columns[1]
    
    print("="*60)
    print("VALIDATION REPORT")
    print(f"Category Column: '{col_category}' | Value Column: '{col_value}'")
    print("="*60)
    
    # Analyze each row
    for idx in df_work.index:
        cat = df_work.at[idx, col_category]
        val = df_work.at[idx, col_value]
        
        # Type checks
        cat_is_str = isinstance(cat, str) or pd.notna(cat)
        
        val_str = str(val)
        
        # Check if value contains percentage
        has_percent = '%' in val_str
        
        # Try numeric conversion
        try:
            # Clean parentheses and percentage signs
            val_clean = val_str.replace('%', '').replace('(', '-').replace(')', '').strip()
            val_num = float(val_clean)
            is_numeric = True
            
            # Range check (allow negative values due to parentheses)
            in_range = -100 <= val_num <= 150
        except:
            is_numeric = False
            in_range = False
            val_num = None
        
        # Determine if row is valid
        is_valid = cat_is_str and is_numeric and in_range
        
        # Print row analysis
        symbol = "✅" if is_valid else "❌"
        print(f"{symbol} Row {idx:2d}: '{str(cat)[:30]:30s}' | '{val_str:8s}' | "
              f"Numeric: {is_numeric} | Range OK: {in_range}")
        
        # Mark for removal if invalid
        if not is_valid:
            df_work.loc[idx, 'INVALID'] = True
    
    print("="*60)
    
    # Remove invalid rows
    if 'INVALID' in df_work.columns:
        df_work = df_work[df_work['INVALID'] != True]
        df_work = df_work.drop('INVALID', axis=1)

    # Final cleaning
    # Only apply string operations if column is not already numeric
    if pd.api.types.is_numeric_dtype(df_work[col_value]):
        df_work[col_value] = df_work[col_value].astype(float)
    else:
        # Clean parentheses (convert to negative) and percentage signs
        df_work[col_value] = df_work[col_value].str.replace('%', '').str.replace('(', '-').str.replace(')', '').astype(float)
    df_work[col_category] = df_work[col_category].str.replace(r'\.{2,}', '', regex=True).str.strip()

    if sort_by_value:
        df_work = df_work.sort_values(by=col_value, ascending=False).reset_index(drop=True)
        print(f"\n✅ Final: {len(df_work)} valid rows (sorted by value)")
    else:
        df_work = df_work.reset_index(drop=True)
        print(f"\n✅ Final: {len(df_work)} valid rows (original order preserved)")
    
    return df_work

import pandas as pd

def process_monthly_data(df, date_column=None):
    """
    Detects and parses monthly data (e.g. 'Jul 15') into real datetime objects.
    """
    if df is None or df.empty: return df
    
    df_clean = df.copy()
    
    # Auto-detect date column
    if date_column is None:
        date_column = df_clean.columns[0]
        
    # Check 1: Is the length sufficient to trigger 'Monthly' logic?
    if len(df_clean) > 9: # Heuristic: >30 rows usually implies monthly/daily
        
        # Check 2: Try converting the first value 'Jul 15' to a date
        first_val = str(df_clean[date_column].iloc[0])
        
        # Skip if first value is a year (e.g., 2014, 2015)
        if re.match(r'^\d{4}$', first_val.strip()):
            return df_clean
            
        try:
            # Try format='%b %y' for "Jul 15", "Aug 15"
            pd.to_datetime(first_val, format='%b %y')
            df_clean[date_column] = pd.to_datetime(df_clean[date_column], format='%b %y')
            print("✅ Detected Monthly Data: Converted 'Jul 15' format to Datetime objects.")
            
        except ValueError:
            # Try format='%b-%y' for "Nov-24", "Dec-24", "Jan-25"
            try:
                pd.to_datetime(first_val, format='%b-%y')
                df_clean[date_column] = pd.to_datetime(df_clean[date_column], format='%b-%y')
                print("✅ Detected Monthly Data: Converted 'Nov-24' format to Datetime objects.")
            except ValueError:
                print("⚠️ Could not parse dates automatically. keeping original.")
                print(df)
            
    return df_clean

def save_and_generate_performance_chart(
    df: pd.DataFrame,
    title: str = "Performance Comparison",
    subtitle: Optional[str] = None,
    output_filename: str = "performance_chart",
    date_column: Optional[str] = None,
    main_series: Optional[str] = None,
    width: int = 900,
    height: int = 600
) -> Tuple[go.Figure, Path]:
    """
    Generate a quarterly performance chart and save it as SVG.
    
    Args:
        df: DataFrame with performance data
        title: Chart title
        subtitle: Optional subtitle
        output_filename: Output filename (without extension)
        date_column: Name of the date column
        main_series: Name of the main series to highlight
        width: Chart width in pixels
        height: Chart height in pixels
    
    Returns:
        Tuple of (Figure object, Path to saved file)
    """
    fig = generate_quarterly_performance_chart(
        df=df,
        title=title,
        subtitle=subtitle,
        date_column=date_column,
        main_series=main_series,
        width=width,
        height=height
    )
    
    path = save_chart_as_svg(
        fig=fig,
        filename=output_filename,
        output_folder="charts/performance",
        width=width,
        height=height
    )
    
    return fig, path


def extract_flexible_performance(df: pd.DataFrame) -> dict:
    """
    Scans a dataframe to find performance headers anywhere in the top rows,
    then extracts the fund's performance from the row immediately following.
    """
    # Initialize result dictionary
    result = {
        "1_year": None,
        "5_year": None, 
        "10_year": None, 
        "since_inception": None
    }
    
    # [VERIFIED] Regex patterns to catch variations like "1 Year", "1 yr", "Since Inception 5/20/20"
    patterns = {
        "1_year": re.compile(r"(?i)\b1\s*y(ea)?r"),
        "5_year": re.compile(r"(?i)\b5\s*y(ea)?r"),
        "10_year": re.compile(r"(?i)\b10\s*y(ea)?r"),
        "since_inception": re.compile(r"(?i)incep")
    }
    
    # Create a search space: Columns + first 5 rows
    # We convert to string to ensure regex works
    search_rows = [df.columns.astype(str).tolist()] + df.head(5).astype(str).values.tolist()
    
    header_map = {}
    header_row_index = -1 # -1 implies the columns are the header
    
    # --- Step 1: Find the Header Row ---
    for i, row in enumerate(search_rows):
        # Check this row against our patterns
        current_map = {}
        for idx, cell_val in enumerate(row):
            for key, pattern in patterns.items():
                if pattern.search(cell_val):
                    current_map[key] = idx
        
        # If we found at least "1_year" (the most common common denominator), we assume this is the header
        if "1_year" in current_map:
            header_map = current_map
            header_row_index = i - 1 # Adjust for 0-based dataframe index (-1 = columns)
            break
    
    if not header_map:
        return result # Could not find headers

    # --- Step 2: Extract Data from the Fund Row ---
    # The fund row is usually the first row *after* the header row.
    # We iterate starting from the row after the header.
    
    start_search_row = header_row_index + 1
    
    for i in range(start_search_row, len(df)):
        row = df.iloc[i]
        
        # [HEURISTIC] Validating this is a data row:
        # Check if the "1_year" column has a valid parseable number/string
        val_to_check = row.iloc[header_map["1_year"]]
        cleaned_check = clean_performance_value(val_to_check)
        
        if cleaned_check is not None:
            # We found the data row! Extract all mapped fields.
            for key, col_idx in header_map.items():
                raw_val = row.iloc[col_idx]
                result[key] = clean_performance_value(raw_val)
            break
            
    return result
    
def clean_performance_value(val):
    """
    Robustly parses financial strings like '(14.24', '14.24%', or '(3.15)%'.
    Handles split columns where closing parenthesis might be missing.
    """
    # [VERIFIED] Basic null checks
    if pd.isna(val) or val == "" or str(val).strip() in ["--", "nan", "None"]:
        return None
    
    val_str = str(val).strip()
    
    # [INFERENCE] In financial tables, '(' usually implies a negative number 
    # even if the closing ')' is cut off by column splitting.
    is_negative = "(" in val_str or val_str.startswith("-")
    
    # [VERIFIED] Regex to extract the first valid floating point number pattern
    # Looks for digits, optionally followed by a dot and more digits
    match = re.search(r"(\d+\.?\d*)", val_str)
    
    if match:
        try:
            number_str = match.group(1)
            float_val = float(number_str)
            
            # Apply negative sign if detected
            return -float_val if is_negative else float_val
        except ValueError:
            return None
            
    return None



def validate_and_clean_performance(df, date_column: str = None):
    """
    Validate and clean performance/time-series DataFrame with multiple value columns.
    Returns cleaned df and prints validation report.
    
    Args:
        df: DataFrame with date/period column and multiple value series columns
        date_column: Name of the date/period column (if None, uses first column or 'Unnamed: 0')
    
    Returns:
        Cleaned DataFrame with date column and numeric value columns
    """
    if df is None or df.empty:
        print("⚠️ Empty DataFrame")
        return None
    

    if len(df) == 36:
        print(df)
    df_work = df.copy().reset_index(drop=True)
    
    # Rename first column to date_column if provided
    if date_column is not None:
        old_first_col = df_work.columns[0]
        df_work = df_work.rename(columns={old_first_col: date_column})
    else:
        # Use the existing first column name
        date_column = df_work.columns[0]
    
    print("="*60)
    print("PERFORMANCE DATA VALIDATION REPORT")
    print(f"Date Column: '{date_column}'")
    print(f"Value Columns: {list(df_work.columns[1:])}")
    print("="*60)
    
    # Identify value columns (all columns except date column)
    value_columns = [col for col in df_work.columns if col != date_column]
    
    # Track invalid rows
    invalid_rows = []
    
    # Validate each row
    for idx in df_work.index:
        date_val = df_work.at[idx, date_column]
        
        # Check if date value is valid
        date_is_valid = pd.notna(date_val) and str(date_val).strip() != ''
        
        # Check each value column
        row_valid = date_is_valid
        numeric_count = 0
        
        for col in value_columns:
            val = df_work.at[idx, col]
            val_str = str(val)
            
            # Try to parse as currency/numeric
            try:
                # Remove currency symbols and commas
                val_clean = val_str.replace('$', '').replace(',', '').strip()
                val_num = float(val_clean)
                numeric_count += 1
            except:
                row_valid = False
                break
        
        # Row is valid if date is valid and all value columns are numeric
        if not row_valid or numeric_count != len(value_columns):
            invalid_rows.append(idx)
            symbol = "❌"
        else:
            symbol = "✅"
        
        # Print summary for first/last few rows and invalid rows
        if idx < 3 or idx >= len(df_work) - 3 or not row_valid:
            print(f"{symbol} Row {idx:2d}: Date='{str(date_val)[:10]:10s}' | Valid values: {numeric_count}/{len(value_columns)}")
    
    print("="*60)
    
    # Remove invalid rows
    if invalid_rows:
        print(f"⚠️ Removing {len(invalid_rows)} invalid rows")
        df_work = df_work.drop(invalid_rows).reset_index(drop=True)
    
    # Clean value columns - convert to numeric
    for col in value_columns:
        df_work[col] = df_work[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    
    # Clean date column
    df_work[date_column] = df_work[date_column].astype(str).str.strip()
    
    # Duplicate first row three times if it's 2014
    if len(df_work) >= 10:
        # CASE A: Monthly Data (Use your helper function)
        print(f"📊 Detected large dataset ({len(df_work)} rows). Attempting Monthly parsing...")
        df_work = process_monthly_data(df_work, date_column)
        
    else:

        df_work[date_column] = df_work[date_column].astype(str).str.strip()
    
        if len(df_work) > 0 and str(df_work[date_column].iloc[0]) == '2014':
            
            # Safety Check: Don't duplicate if already done
            is_already_duplicated = (len(df_work) > 1) and df_work.iloc[0].equals(df_work.iloc[1])
            
            if not is_already_duplicated:
                first_row = df_work.iloc[0:1].copy()
                df_work = pd.concat([first_row, first_row.copy(), first_row.copy(), df_work], ignore_index=True)
                print(f"🔄 Added 3 duplicate rows of 2014 for better visualization")

   
    df_work['_x_index'] = range(len(df_work))
    
    print(f"\n✅ Final: {len(df_work)} valid rows with {len(value_columns)} value series")
    print(f"Date range: {df_work[date_column].iloc[0]} to {df_work[date_column].iloc[-1]}")
    
    return df_work


def generate_performance_line_chart(
    df: pd.DataFrame,
    title: str = "Performance Comparison",
    subtitle: Optional[str] = None,
    date_column: Optional[str] = None,
    main_series: Optional[str] = None,
    width: int = 900,
    height: int = 600,
    show_legend: bool = True,
    legend_position: str = "top"
) -> go.Figure:
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Auto-detect date column
    if date_column is None:
        date_column = df.columns[0]
        
    # --- MODIFICATION 1: Safety Check for _x_index ---
    # If the cleaning function wasn't run, create the index here to prevent crash
    if '_x_index' not in df.columns:
        df = df.copy()
        df['_x_index'] = range(len(df))

    # Get value columns (all except date column and index)
    value_columns = [col for col in df.columns if col not in [date_column, '_x_index']]
    
    # Auto-detect main series
    if main_series is None:
        main_series = value_columns[0]
    
    # Define color palette
    main_color = '#0A2463'
    benchmark_colors = ['#FFD700', '#FFA500', '#00CED1']
    fig = go.Figure()
    
    # Use sequential index for x-axis
    x_values = df['_x_index']
    
    # Add traces
    for idx, col in enumerate(value_columns):
        is_main = (col == main_series)
        
        if is_main:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=main_color, width=6, shape='spline'),
                fill='tozeroy',
                fillcolor='rgba(46, 94, 170, 0.15)',
                # --- MODIFICATION 2: Use "customdata" for Hover ---
                # We pass the REAL date string to customdata so the tooltip shows "2020" not "0"
                customdata=df[date_column],
                hovertemplate=f'<b>{col}</b><br>%{{customdata}}<br>${{y:,.0f}}<extra></extra>'
            ))
        else:
            color_idx = (idx - 1) % len(benchmark_colors) if is_main else idx % len(benchmark_colors)
            fig.add_trace(go.Scatter(
                x=x_values,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=benchmark_colors[color_idx], width=2, shape='spline', dash='dash'),
                
                customdata=df[date_column]
            ))
    
    # --- MODIFICATION 3: Smart Tick Selection ---
    # Show year label only on its FIRST occurrence (avoids duplicate labels)
    tick_df = df.drop_duplicates(subset=[date_column], keep='first')
    
    # These are the exact positions where the year label will appear
    final_tickvals = tick_df['_x_index'].tolist()
    final_ticktext = tick_df[date_column].astype(str).tolist()

    # Layout Setup (Same as before but with specific tick lists)
    if subtitle:
        title_text = f"<b>{title}</b><br><sub>{subtitle}</sub>"
    else:
        title_text = f"<b>{title}</b>"

    # (Legend config logic remains the same...)
    if legend_position == "top":
        legend_config = dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5)
    elif legend_position == "bottom":
        legend_config = dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    else:
        legend_config = dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color='white', family='Arial, sans-serif')
        ),
        showlegend=show_legend,
        legend=dict(
            **legend_config,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#444',
            borderwidth=1,
            font=dict(size=13, color='white'),
            itemsizing='constant',
            itemwidth=30
        ),
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11, family='Arial, sans-serif', color='white'),
        margin=dict(l=80, r=80, t=250, b=80),
        
        xaxis=dict(
            title=dict(text=date_column, font=dict(size=13, color='white'), standoff=15),
            gridcolor='#2d2d2d',
            showgrid=True,
            linecolor='#444',
            linewidth=1,
            
            # --- MODIFICATION 4: Apply the Clean Ticks ---
            tickmode='array',
            tickvals=final_tickvals, # Only the 15 selected points
            ticktext=final_ticktext, # Only the 15 selected labels
            tickfont=dict(size=11, color='white')
        ),
        yaxis=dict(
            title=dict(text="Value ($)", font=dict(size=13, color='white'), standoff=15),
            gridcolor='#2d2d2d',
            showgrid=True,
            tickfont=dict(size=11, color='white'),
            tickformat='$,.0f',
            linecolor='#444',
            linewidth=1
        ),
        hovermode='x unified'
    )
    
    return fig


def generate_quarterly_performance_chart(
    df: pd.DataFrame,
    title: str = "Performance Comparison",
    subtitle: Optional[str] = None,
    date_column: Optional[str] = None,
    main_series: Optional[str] = None,
    width: int = 900,
    height: int = 600,
    show_legend: bool = True,
    legend_position: str = "top"
) -> go.Figure:
    
    if df is None or df.empty: raise ValueError("DataFrame is empty")
    
    df_plot = df.copy()
    if date_column is None: date_column = df_plot.columns[0]

    # --- YEAR EXTRACTION ---
    try:
        temp_dates = pd.to_datetime(df_plot[date_column], errors='coerce')
        if temp_dates.notna().all():
            df_plot['_group_year'] = temp_dates.dt.year.astype(str)
        else:
            df_plot['_group_year'] = df_plot[date_column].astype(str).str[:4]
    except:
        df_plot['_group_year'] = df_plot[date_column].astype(str)

    if '_x_index' not in df_plot.columns: df_plot['_x_index'] = range(len(df_plot))

    value_columns = [col for col in df_plot.columns if col not in [date_column, '_x_index', '_group_year']]
    if main_series is None: main_series = value_columns[0]
    
    # Colors
    main_color = '#0A2463'
    benchmark_colors = ['#FFD700', '#FFA500', '#00CED1']
    
    fig = go.Figure()
    
    # PLOT TRACES
    for idx, col in enumerate(value_columns):
        is_main = (col == main_series)
        fig.add_trace(go.Scatter(
            x=df_plot['_x_index'], y=df_plot[col],
            mode='lines', name=col,
            line=dict(
                color=main_color if is_main else benchmark_colors[idx % len(benchmark_colors)], 
                width=5 if is_main else 1,
                shape='spline', dash='solid' if is_main else 'dash'
                
            ),
            opacity=1.0 if is_main else 0.5,
            fill='tozeroy' if is_main else None, 
            fillcolor='rgba(10, 36, 99, 0.15)',
            customdata=df_plot[date_column],
            hovertemplate=f'<b>{col}</b><br>%{{customdata}}<br>${{y:,.0f}}<extra></extra>'
        ))
    
    # --- SMART TICK GENERATION (The Fix) ---
    tick_vals = []
    tick_text = []
    
    # 1. Decide on density based on total row count
    total_points = len(df_plot)
    
    # If we have many points, ONLY show the Year label (skip quarters)
    show_quarters = total_points < 10 
    
    grouped = df_plot.groupby('_group_year', sort=False)
    
    for year_label, group in grouped:
        indices = group['_x_index'].tolist()
        
        for i, real_idx in enumerate(indices):
            # i=0 is the first point of the year (Q1 or Jan)
            
            if i == 0:
                # ALWAYS show the Year label
                # If we are showing quarters, add "Q1" context, otherwise just "2020"
                if show_quarters:
                    label = f"{year_label} Q1"
                else:
                    label = year_label # Just "2015", "2016"... clean and simple
                
                tick_vals.append(real_idx)
                tick_text.append(label)
                
            elif show_quarters:
                # Only add sub-labels (Q2, Q3) if we have enough space
                q_num = i + 1
                label = f"Q{q_num}"
                tick_vals.append(real_idx)
                tick_text.append(label)

    # --- LAYOUT ---
    if subtitle: title_text = f"<b>{title}</b><br><sub>{subtitle}</sub>"
    else: title_text = f"<b>{title}</b>"

    if legend_position == "top": legend_config = dict(orientation="h", y=1.08, x=0.5, xanchor="center")
    elif legend_position == "bottom": legend_config = dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
    else: legend_config = dict(orientation="v", y=0.5, x=1.02, xanchor="left")

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=20, color='white')),
        showlegend=show_legend,
        legend=dict(**legend_config, font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
        width=width, height=height,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=60, r=40, t=200, b=80),
        
        xaxis=dict(
            title=dict(text="Period", font=dict(size=13, color='white'), standoff=10),
            gridcolor='#2d2d2d',
            linecolor='#444',
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=11, color='white'),
            tickangle=45 
        ),
        yaxis=dict(
            title=dict(text="Value ($)", font=dict(color='white')),
            gridcolor='#2d2d2d',
            tickformat='$,.0f'
        ),
        hovermode='x unified'
    )
    
    return fig


def generate_portfolio_pie_chart(
    df: pd.DataFrame,
    title: str = "Portfolio Composition",
    subtitle: Optional[str] = None,
    col_category: Optional[str] = None,
    col_value: Optional[str] = None,
    min_value_threshold: float = 0.5,
    width: int = 600,
    height: int = 450,
    hole_size: float = 0.35,
    pull_largest: bool = True,
    color_scheme: str = "Turbo"
) -> go.Figure:
    """
    Generate a modern dark-themed donut pie chart for portfolio composition.
    
    Args:
        df: DataFrame with category and value columns
        title: Chart main title (e.g., fund name)
        subtitle: Chart subtitle (e.g., "Portfolio Composition")
        col_category: Name of category column (if None, uses first column)
        col_value: Name of value column (if None, uses second column)
        min_value_threshold: Minimum value to include in chart (filters small slices)
        width: Chart width in pixels
        height: Chart height in pixels
        hole_size: Size of donut hole (0 = full pie, 0.5 = half donut)
        pull_largest: Whether to pull out the largest slice
        color_scheme: Plotly color scheme (Viridis, Plasma, Inferno, etc.)
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> df = pd.DataFrame({'Category': ['Stocks', 'Bonds'], 'Value': [60, 40]})
        >>> fig = generate_portfolio_pie_chart(df, title="My Fund", subtitle="Portfolio Composition")
        >>> fig.show()
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Auto-detect columns if not specified
    if col_category is None:
        col_category = df.columns[0]
    if col_value is None:
        col_value = df.columns[1]
    
    # Filter out small or negative values
    df_positive = df[df[col_value] > min_value_threshold].copy()
    
    if df_positive.empty:
        raise ValueError(f"No values above threshold {min_value_threshold}")
    
    # Sort by value descending (largest first)
    df_positive = df_positive.sort_values(by=col_value, ascending=False).reset_index(drop=True)
    
    # Truncate long category names for display
    max_label_length = 45
    df_positive['display_label'] = df_positive[col_category].apply(
        lambda x: x if len(str(x)) <= max_label_length else str(x)[:max_label_length] + '...'
    )
    df_positive['full_label'] = df_positive[col_category]  # Keep original for hover
    
    # Get color scheme
    # Use distinct colors for small datasets (< 6 items)
    if len(df_positive) < 6:
        # Distinct, easily differentiable colors
        distinct_colors = [
            '#3B82F6',  # Bright Blue
            '#10B981',  # Emerald Green
            '#F59E0B',  # Amber
            '#EF4444',  # Red
            '#8B5CF6',  # Purple
        ]
        hole_size = 0.30
        colors = distinct_colors[:len(df_positive)]
    else:
        # Use standard color schemes for larger datasets
        distinct_colors = [
            '#3B82F6',  # Bright Blue
            '#10B981',  # Emerald Green
            '#F59E0B',  # Amber (Yellow-Orange)
            '#EF4444',  # Red
            '#8B5CF6',  # Purple
            
            # --- EXTENDED VIVID COLORS ---
            '#EC4899',  # Hot Pink (High contrast vs Dark)
            '#06B6D4',  # Cyan (Bright & Electric)
            '#84CC16',  # Lime (Acid Green - distinct from Emerald)
            '#F97316',  # Deep Orange (Distinct from Amber)
            '#6366F1',  # Indigo (Deep Blue-Purple)
            '#14B8A6',  # Teal (Blue-Green mix)
        ]
        colors = distinct_colors[:len(df_positive)]
    
    # Calculate pull values
    if pull_largest:
        pull_values = [0.05 if i == 0 else 0 for i in range(len(df_positive))]
    else:
        pull_values = [0] * len(df_positive)
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=df_positive['display_label'],  # Use truncated labels
        values=df_positive[col_value],
        hole=hole_size,
        marker=dict(
            colors=colors,
            line=dict(color='#1e1e1e', width=8)
        ),
        textposition='outside',  # Auto positioning to prevent overlap
        textinfo='label+percent',
        textfont=dict(size=9, color='white', family='Arial, sans-serif'),
        hovertemplate='<b>%{customdata}</b><br>%{value:.1f}%<br>%{percent}<extra></extra>',
        customdata=df_positive['full_label'],  # Full name in hover
        direction='clockwise',  # Clockwise arrangement
        rotation=200,  # Start from top, larger values go right
        sort=False  # Keep our sorted order (largest first)
    )])
    
    # Update layout
    # Format title with subtitle
    if subtitle:
        title_text = f"<b>{title}</b><br><sub>{subtitle}</sub>"
    else:
        title_text = f"<b>{title}</b>"
    
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color='white', family='Arial, sans-serif'),
            pad=dict(b=20)  # Add bottom padding to title for more space
        ),
        showlegend=False,
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size = 9, family='Arial, sans-serif', color='white'),
        margin=dict(l=50, r=50, t=130, b=50),  # Increased top margin for more space between title and chart
        uniformtext=dict(minsize=7, mode='hide')
    )
    
    return fig


def generate_portfolio_bar_chart(
    df: pd.DataFrame,
    title: str = "Portfolio Analysis",
    subtitle: Optional[str] = None,
    col_category: Optional[str] = None,
    col_value: Optional[str] = None,
    width: int = 700,
    height: int = 500,
    orientation: str = 'h',
    show_values: bool = True,
    sort_by_value: bool = True,
    sort_alphabetically: bool = False,
    color_scheme: str = "intelligent"
) -> go.Figure:
    """
    Generate a modern dark-themed bar chart for portfolio data with support for negative values.
    
    Args:
        df: DataFrame with category and value columns
        title: Chart main title (e.g., fund name)
        subtitle: Chart subtitle (e.g., "Performance Analysis")
        col_category: Name of category column (if None, uses first column)
        col_value: Name of value column (if None, uses second column)
        width: Chart width in pixels
        height: Chart height in pixels
        orientation: 'h' for horizontal bars, 'v' for vertical bars
        show_values: Whether to show value labels on bars
        sort_by_value: Whether to sort bars by value (descending). Ignored if sort_alphabetically is True.
        sort_alphabetically: If True, sort by category column alphabetically. Takes precedence over sort_by_value.
        color_scheme: 'intelligent' (smart colors based on value ranges), 'diverging' (red/green for neg/pos), 'category' (different color per bar), 'gradient', or 'uniform'
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> df = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Value': [10, -5, 15]})
        >>> fig = generate_portfolio_bar_chart(df, title="Performance", subtitle="Returns")
        >>> fig.show()
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Auto-detect columns if not specified
    if col_category is None:
        col_category = df.columns[0]
    if col_value is None:
        col_value = df.columns[1]
    
    # Create working copy
    df_work = df.copy()
    
    # Sort data
    if sort_alphabetically:
        # Sort alphabetically by category column
        df_work = df_work.sort_values(by=col_category, ascending=False if orientation == 'h' else False)
    elif sort_by_value:
        # Sort by value if requested
        df_work = df_work.sort_values(by=col_value, ascending=True if orientation == 'h' else False)
    
    # Truncate long category names for display
    max_label_length = 30 if orientation == 'h' else 20  # Reduced from 40 to 30
    df_work['display_label'] = df_work[col_category].apply(
        lambda x: x if len(str(x)) <= max_label_length else str(x)[:max_label_length] + '...'
    )
    df_work['full_label'] = df_work[col_category]  # Keep original for hover
    
    # Determine colors based on scheme
    if color_scheme == 'diverging':
        # Red for negative, green for positive
        colors = [
            '#EF4444' if val < 0 else '#10B981'  # Red or Green
            for val in df_work[col_value]
        ]
    elif color_scheme == 'gradient':
        # Use a gradient based on value magnitude
        colors = px.colors.sequential.Viridis
    elif color_scheme == 'intelligent':
        # Intelligent color selection based on value ranges
        colors = []
        for val in df_work[col_value]:
            if val < 0:
                # Negative values: darker red for more negative
                if val < -5:
                    colors.append('#DC2626')  # Dark Red
                elif val < -2:
                    colors.append('#EF4444')  # Red
                else:
                    colors.append('#F87171')  # Light Red
            elif val < 5:
                # Low positive values: Blue tones
                colors.append('#3B82F6')  # Blue
            elif val < 15:
                # Medium values: Teal/Cyan
                colors.append('#14B8A6')  # Teal
            elif val < 30:
                # Good values: Green tones
                colors.append('#10B981')  # Emerald
            else:
                # Excellent values: Bright green
                colors.append('#22C55E')  # Bright Green
    # Create bar chart
    if orientation == 'h':
        fig = go.Figure(data=[go.Bar(
            x=df_work[col_value],
            y=df_work['display_label'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#1e1e1e', width=1),
                cornerradius=8  # Rounded corners for bars
            ),
            text=df_work[col_value].apply(lambda x: f'{x:.1f}%') if show_values else None,
            textposition='outside',  # Changed from 'outside' to 'auto' for better positioning
            textfont=dict(size=10, color='white', family='Arial, sans-serif'),
            hovertemplate='<b>%{customdata}</b><br>Value: %{x:.2f}%<extra></extra>',
            customdata=df_work['full_label'],
            cliponaxis=False  # Prevent text from being clipped
        )])
        
        # Add vertical line at x=0 for reference
        fig.add_vline(x=0, line_width=2, line_color='white', opacity=0.5)
    
    # Format title with subtitle
    if subtitle:
        title_text = f"<b>{title}</b><br><sub>{subtitle}</sub>"
    else:
        title_text = f"<b>{title}</b>"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color='white', family='Arial, sans-serif'),
            pad=dict(l=25)
        ),
        showlegend=False,
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11, family='Arial, sans-serif', color='white'),
        #margin=dict(l=250 if orientation == 'h' else 80, r=100, t=100, b=80),  # Increased left margin to prevent label overlap
        xaxis=dict(
            gridcolor='#2d2d2d',
            showgrid=True,
            zeroline=False,
            title=dict(
                text=col_value if orientation == 'h' else col_category,
                font=dict(size=13, color='white'),
                standoff=35
            ),
            tickfont=dict(size=12, color='white'),
            dtick=20  # Set tick interval to 10 (shows 10, 20, 30, 40, etc.)
        ),
        yaxis=dict(
            automargin=True,
            gridcolor='#2d2d2d',
            showgrid=True if orientation == 'v' else False,
            zeroline=False,
            title=dict(
                text=col_category if orientation == 'h' else col_value,
                font=dict(size=13, color='white'),
                standoff=40  # Increase space between y-axis title and tick labels
            ),
            tickfont=dict(size=11, color='white'),  # Y-axis category label size
            ticklen=4,  # Length of tick marks
            ticklabelstandoff=15   # Space between tick labels and the plot area
        ),
        bargap=0.3,  # Increased gap = thinner bars (0.3 = 30% gap between bars)
        uniformtext=dict(minsize=8, mode='hide')
    )
    
    return fig

def generate_formal_top_holdings_chart(
    df: pd.DataFrame,
    title: str = "Top 10 Holdings",
    subtitle: Optional[str] = "Portfolio Weight",
    col_name: Optional[str] = None,
    col_value: Optional[str] = None,
    top_n: int = 10,
    width: int = 800,
    height: int = 500,
    max_label_len: int = 30  # Character limit for names
) -> go.Figure:
    """
    Generate a formal, Morningstar-style horizontal bar chart.
    Designed to fit 10 values perfectly with smart text handling.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")

    # 1. Setup Columns
    if col_name is None: col_name = df.columns[0]
    if col_value is None: col_value = df.columns[1]

    # 2. Process Data (Top N)
    df_top = df.head(top_n).copy()
    df_top[col_value] = pd.to_numeric(df_top[col_value], errors='coerce')
    
    # Sort Ascending so the Largest bar is at the Top visual position
    df_top = df_top.sort_values(by=col_value, ascending=True)

    # 3. Handle Long Text (Truncation)
    # Creates "Zoom Video Comm..." while keeping full name for hover
    df_top['display_name'] = df_top[col_name].apply(
        lambda x: x if len(str(x)) <= max_label_len else str(x)[:max_label_len].strip() + "..."
    )

    # 4. Create the Chart
    fig = go.Figure(go.Bar(
        x=df_top[col_value],
        y=df_top['display_name'],
        orientation='h',
        
        # HOVER: Shows the full untruncated name
        customdata=df_top[col_name],
        hovertemplate='<b>%{customdata}</b><br>Weight: %{x:.2f}%<extra></extra>',

        # VISUALS: Professional Blue
        marker=dict(
            color='#4472C4',  # Standard "Financial Report" Blue
            line=dict(width=0),
            cornerradius=0    # Sharp corners look more formal/traditional
        ),

        # TEXT LABELS (The Values)
        text=df_top[col_value].apply(lambda x: f"{x:.2f}%"),
        textposition='outside', # Places numbers at the end of the bar
        textfont=dict(
            family='Arial', 
            size=12, 
            color='white'
        ),
        
        # Ensure bars are not too thick (0.6 - 0.7 is elegant)
        width=0.6,
        cliponaxis=False 
    ))

    # 5. Formal Layout Configuration
    if subtitle:
        title_text = f"<b>{title.upper()}</b><br><span style='font-size:13px;color:#666666'>{subtitle}</span>"
    else:
        title_text = f"<b>{title.upper()}</b>"

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.0, # Left aligned
            y=0.95,
            font=dict(family='Arial', size=18, color='white')
        ),
        
        # White Background (Formal Print Style)
        paper_bgcolor='rgba(0,0,0,0)',  
        plot_bgcolor='rgba(0,0,0,0)',  
        
        width=width,
        height=height,

        xaxis=dict(
            showgrid=True,       
            gridcolor='#9E9E9E',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='#9E9E9E',
            showticklabels=True, # Show numbers on bottom axis
            tickfont=dict(size=10, color='white'),
            side='top'           # OPTIONAL: Put X-axis on top like some reports
        ),
        
        yaxis=dict(
            showgrid=False,
            showline=False,
            tickfont=dict(family='Arial', size=12, color='white'),
            ticklen=5,
            ticklabelstandoff=15
        ),
        
        bargap=0.3 # Space between bars
    )

    return fig



def save_chart_as_svg(
    fig: go.Figure,
    filename: str,
    output_folder: str = "charts",
    width: int = 800,
    height: int = 500
) -> Path:
    """
    Save a Plotly figure as SVG file.
    
    Args:
        fig: Plotly Figure object
        filename: Output filename (without extension)
        output_folder: Folder to save the chart (created if doesn't exist)
        width: Output width in pixels
        height: Output height in pixels
    
    Returns:
        Path object pointing to saved file
    
    Raises:
        ImportError: If kaleido is not installed
        
    Example:
        >>> fig = generate_portfolio_pie_chart(df)
        >>> path = save_chart_as_svg(fig, "my_portfolio")
        >>> print(f"Saved to: {path}")
    """
    try:
        # Create output folder
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Clean filename
        clean_filename = filename.replace(' ', '_').replace('/', '_')
        if not clean_filename.endswith('.svg'):
            clean_filename += '.svg'
        
        # Full path
        full_path = output_path / clean_filename
        
        # Save as SVG
        fig.write_image(full_path, width=width, height=height, engine="kaleido")
        
        return full_path
        
    except ImportError as e:
        raise ImportError(
            "Kaleido is required to save static images. Install it with:\n"
            "  pip install kaleido\n"
            "or\n"
            "  conda install -c conda-forge python-kaleido"
        ) from e

def generate_and_save_holdings_chart(
    df: pd.DataFrame,
    title: str,
    output_filename: str,
    subtitle: Optional[str] = "Top Holdings",
    output_folder: str = "charts/holdings",
    **kwargs
) -> Tuple[go.Figure, Path]:

    fig = generate_formal_top_holdings_chart(df, title, subtitle, **kwargs)
    
    path = save_chart_as_svg(fig, output_filename, output_folder)
    return fig, path

def generate_and_save_portfolio_chart(
    df: pd.DataFrame,
    title: str,
    pie: bool,
    output_filename: str,
    subtitle: Optional[str] = "Portfolio Composition",
    output_folder: str = "charts/portfolio",
    **kwargs
) -> Tuple[go.Figure, Path]:
    """
    Convenience function to generate and save a portfolio chart in one call.
    
    Args:
        df: DataFrame with portfolio data
        title: Chart main title (e.g., fund name)
        output_filename: Output filename (without extension)
        subtitle: Chart subtitle (default: "Portfolio Composition")
        output_folder: Folder to save the chart
        **kwargs: Additional arguments passed to generate_portfolio_pie_chart
    
    Returns:
        Tuple of (Figure object, Path to saved file)
    
    Example:
        >>> fig, path = generate_and_save_portfolio_chart(
        ...     df, 
        ...     title="My Fund",
        ...     subtitle="Portfolio Composition",
        ...     output_filename="my_fund_portfolio"
        ... )
        >>> print(f"Chart saved to: {path}")
        >>> fig.show()
    """
    # Generate chart
    if pie:
        fig = generate_portfolio_pie_chart(df, title=title, subtitle=subtitle,width=850, height=550, hole_size=0.2, **kwargs)
    else:
        fig = generate_portfolio_bar_chart(df, title=title, subtitle=subtitle, **kwargs)
    
    # Save chart
    path = save_chart_as_svg(fig, output_filename, output_folder)
    
   
    
    return fig, path



def generate_top_holdings_table(df):
        
    # Get top 10 rows ordered by percent in descending order
    df_top10 = df.nlargest(10, 'percent')

    # Create a formatted version for display
    # Get top 10 rows ordered by percent in descending order
    df_top10 = df.nlargest(10, 'percent')

    # Calculate totals BEFORE formatting
    total_value = df_top10['value_usd'].sum()
    total_percent = df_top10['percent'].sum()
    total_balance = df_top10['balance'].astype(float).sum()

    # Create a formatted version for display
    df_markdown = df_top10.copy()

    # Format the columns
    df_markdown['value_usd'] = df_markdown['value_usd'].apply(lambda x: f'${x:,.2f}')
    df_markdown['percent'] = df_markdown['percent'].apply(lambda x: f'{x/10:.4f}%')
    df_markdown['balance'] = df_markdown['balance'].astype(float).apply(lambda x: f'{x:,.2f}')

    # Rename columns
    df_markdown = df_markdown.rename(columns={
        'name': 'Company Name',
        'value_usd': 'Value (USD)',
        'percent': 'Percentage',
        'asset_type': 'Asset Type',
        'country': 'Country',
        'balance': 'Balance',
        'sector': 'Sector'
    })

    # Reorder columns
    df_markdown = df_markdown[['Company Name', 'Percentage', 'Value (USD)', 'Balance', 'Sector', 'Asset Type', 'Country']]

    # Reset index and add rank
    df_markdown = df_markdown.reset_index(drop=True)
    df_markdown.insert(0, 'Rank', range(1, len(df_markdown) + 1))

    # Replace NaN with dash
    df_markdown = df_markdown.fillna('-')

    # Add total row
    total_row = pd.DataFrame([{
        'Rank': 'TOTAL',
        'Company Name': f'{len(df_markdown)} Holdings',
        'Percentage': f'{total_percent/10:.4f}%',
        'Value (USD)': f'${total_value:,.2f}',
        'Balance': f'{total_balance:,.2f}',
        'Sector': '-',
        'Asset Type': '-',
        'Country': '-'
    }])

    df_markdown = pd.concat([df_markdown, total_row], ignore_index=True)

    # With styling
    df_markdown.style.set_properties(**{
        'text-align': 'left',
        'font-size': '11pt'
    }).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0'), ('text-align', 'left')]},
        {'selector': 'td:nth-child(1)', 'props': [('font-weight', 'bold')]},  # Bold rank column
        {'selector': 'tr:last-child td', 'props': [('font-weight', 'bold')]}  # Bold last row
    ]).hide(axis='index')

    return df_markdown

# Alternative save functions for different formats
def save_chart_as_png(
    fig: go.Figure,
    filename: str,
    output_folder: str = "charts",
    width: int = 600,
    height: int = 450,
    scale: float = 2.0
) -> Path:
    """
    Save chart as PNG (higher quality, larger file size).
    
    Args:
        scale: Resolution multiplier (2.0 = retina quality)
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    clean_filename = filename.replace(' ', '_').replace('/', '_')
    if not clean_filename.endswith('.png'):
        clean_filename += '.png'
    
    full_path = output_path / clean_filename
    fig.write_image(full_path, width=width, height=height, scale=scale, engine="kaleido")
    
    return full_path


def save_chart_as_webp(
    fig: go.Figure,
    filename: str,
    output_folder: str = "charts",
    width: int = 600,
    height: int = 450,
    scale: float = 1.5
) -> Path:
    """
    Save chart as WebP (smallest file size, good quality).
    
    Args:
        scale: Resolution multiplier (1.5 = good balance)
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    clean_filename = filename.replace(' ', '_').replace('/', '_')
    if not clean_filename.endswith('.webp'):
        clean_filename += '.webp'
    
    full_path = output_path / clean_filename
    fig.write_image(full_path, width=width, height=height, scale=scale, engine="kaleido")
    
    return full_path


def save_chart_as_html(
    fig: go.Figure,
    filename: str,
    output_folder: str = "charts",
    include_plotlyjs: str = 'cdn'
) -> Path:
    """
    Save chart as interactive HTML.
    
    Args:
        include_plotlyjs: 'cdn' (smallest), 'directory', or True (self-contained)
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    clean_filename = filename.replace(' ', '_').replace('/', '_')
    if not clean_filename.endswith('.html'):
        clean_filename += '.html'
    
    full_path = output_path / clean_filename
    fig.write_html(
        full_path,
        include_plotlyjs=include_plotlyjs,
        config={'displayModeBar': False}
    )
    
    return full_path
