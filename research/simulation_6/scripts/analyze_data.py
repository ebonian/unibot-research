"""
Data Quality Analysis Script for Simulation 6 Parquet Data
Analyzes completeness, correctness, and generates visualizations
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Configuration
DATA_DIR = Path("/Users/ohm/Documents/GitHub/ice-senior-project/research/simulation_6/downloaded_data/daily")
OUTPUT_DIR = Path("/Users/ohm/Documents/GitHub/ice-senior-project/research/simulation_6")

def get_date_range_from_files(folder_path):
    """Extract date range from parquet file names"""
    dates = []
    if not folder_path.exists():
        return None, None, []
    
    for f in sorted(folder_path.glob("*.parquet")):
        try:
            date_str = f.stem  # e.g., "2025-11-03"
            dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            pass
    
    if not dates:
        return None, None, []
    return min(dates), max(dates), sorted(dates)


def check_missing_dates(dates, start_date, end_date):
    """Find missing dates in the range"""
    if not dates:
        return []
    
    all_dates = set()
    current = start_date
    while current <= end_date:
        all_dates.add(current)
        current += timedelta(days=1)
    
    return sorted(all_dates - set(dates))


def analyze_parquet_files(folder_path, category_name):
    """Analyze all parquet files in a folder"""
    results = {
        "category": category_name,
        "file_count": 0,
        "total_rows": 0,
        "columns": [],
        "date_range": None,
        "missing_dates": [],
        "file_sizes": [],
        "empty_files": [],
        "null_counts": {},
        "sample_data": None,
        "daily_row_counts": {},
        "errors": []
    }
    
    if not folder_path.exists():
        results["errors"].append(f"Folder does not exist: {folder_path}")
        return results
    
    files = list(folder_path.glob("*.parquet"))
    results["file_count"] = len(files)
    
    if not files:
        results["errors"].append("No parquet files found")
        return results
    
    start_date, end_date, dates = get_date_range_from_files(folder_path)
    if start_date and end_date:
        results["date_range"] = f"{start_date} to {end_date}"
        results["missing_dates"] = check_missing_dates(dates, start_date, end_date)
    
    for f in sorted(files):
        try:
            size_kb = f.stat().st_size / 1024
            results["file_sizes"].append((f.name, size_kb))
            
            df = pd.read_parquet(f)
            row_count = len(df)
            results["total_rows"] += row_count
            
            date_str = f.stem
            results["daily_row_counts"][date_str] = row_count
            
            if row_count == 0:
                results["empty_files"].append(f.name)
            
            if not results["columns"]:
                results["columns"] = list(df.columns)
            
            # Track null counts
            for col in df.columns:
                if col not in results["null_counts"]:
                    results["null_counts"][col] = 0
                results["null_counts"][col] += df[col].isnull().sum()
            
            # Store sample data from first file
            if results["sample_data"] is None and len(df) > 0:
                results["sample_data"] = df.head(3)
                
        except Exception as e:
            results["errors"].append(f"Error reading {f.name}: {str(e)}")
    
    return results


def plot_daily_row_counts(all_results):
    """Create a visualization of daily row counts for all categories"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    categories = [r for r in all_results if r["file_count"] > 0]
    
    for idx, result in enumerate(categories[:6]):
        ax = axes[idx]
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in result["daily_row_counts"].keys()]
        counts = list(result["daily_row_counts"].values())
        
        if dates and counts:
            ax.bar(dates, counts, width=0.8, alpha=0.7, color='steelblue')
            ax.set_title(f'{result["category"].upper()}\nTotal: {result["total_rows"]:,} rows')
            ax.set_xlabel('Date')
            ax.set_ylabel('Row Count')
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add mean line
            if counts:
                mean_val = sum(counts) / len(counts)
                ax.axhline(y=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.0f}')
                ax.legend()
    
    # Hide unused axes
    for idx in range(len(categories), 6):
        axes[idx].set_visible(False)
    
    plt.suptitle('Daily Row Counts by Data Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'daily_row_counts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: daily_row_counts.png")


def plot_file_sizes(all_results):
    """Create a visualization of file sizes"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    categories = []
    sizes = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, result in enumerate(all_results):
        if result["file_sizes"]:
            for fname, size in result["file_sizes"]:
                categories.append(result["category"][:6])
                sizes.append(size)
    
    # Create box plot by category
    data_by_category = {}
    for r in all_results:
        if r["file_sizes"]:
            data_by_category[r["category"]] = [s for _, s in r["file_sizes"]]
    
    if data_by_category:
        ax.boxplot(data_by_category.values(), labels=data_by_category.keys())
        ax.set_ylabel('File Size (KB)')
        ax.set_title('File Size Distribution by Category')
        ax.set_xlabel('Category')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'file_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: file_sizes.png")


def plot_data_completeness(all_results):
    """Visualize data completeness"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: File count by category
    categories = [r["category"] for r in all_results]
    file_counts = [r["file_count"] for r in all_results]
    
    colors = plt.cm.Set3(range(len(categories)))
    bars = axes[0].bar(categories, file_counts, color=colors)
    axes[0].set_title('File Count by Category')
    axes[0].set_ylabel('Number of Files')
    axes[0].set_xlabel('Category')
    
    # Add value labels
    for bar, count in zip(bars, file_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    # Right: Total rows by category
    total_rows = [r["total_rows"] for r in all_results]
    bars = axes[1].bar(categories, total_rows, color=colors)
    axes[1].set_title('Total Rows by Category')
    axes[1].set_ylabel('Total Rows')
    axes[1].set_xlabel('Category')
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Add value labels
    for bar, count in zip(bars, total_rows):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02, 
                    f'{count:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_completeness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: data_completeness.png")


def plot_missing_dates_heatmap(all_results):
    """Create a heatmap showing data availability"""
    # Collect all dates across all categories
    all_dates = set()
    for r in all_results:
        for date_str in r["daily_row_counts"].keys():
            all_dates.add(date_str)
    
    if not all_dates:
        print("⚠️ No dates to plot for heatmap")
        return
    
    all_dates = sorted(all_dates)
    categories = [r["category"] for r in all_results]
    
    # Create matrix
    matrix = []
    for r in all_results:
        row = []
        for d in all_dates:
            if d in r["daily_row_counts"]:
                row.append(1 if r["daily_row_counts"][d] > 0 else 0.5)
            else:
                row.append(0)
        matrix.append(row)
    
    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn')
    
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    
    # Show every 7th date
    step = max(1, len(all_dates) // 15)
    ax.set_xticks(range(0, len(all_dates), step))
    ax.set_xticklabels([all_dates[i] for i in range(0, len(all_dates), step)], rotation=45, ha='right')
    
    ax.set_title('Data Availability Heatmap (Green = Data Present, Red = Missing)')
    plt.colorbar(im, ax=ax, label='Data Available', shrink=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_availability_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: data_availability_heatmap.png")


def print_analysis_report(all_results):
    """Print detailed analysis report"""
    print("\n" + "="*80)
    print("📊 DATA QUALITY ANALYSIS REPORT")
    print("="*80)
    
    for result in all_results:
        print(f"\n{'─'*60}")
        print(f"📁 {result['category'].upper()}")
        print(f"{'─'*60}")
        print(f"  Files: {result['file_count']}")
        print(f"  Total Rows: {result['total_rows']:,}")
        print(f"  Date Range: {result['date_range']}")
        
        if result['missing_dates']:
            print(f"  ⚠️  Missing Dates ({len(result['missing_dates'])}): {result['missing_dates'][:5]}...")
        else:
            print(f"  ✅ No missing dates")
        
        if result['empty_files']:
            print(f"  ⚠️  Empty Files ({len(result['empty_files'])}): {result['empty_files'][:5]}")
        else:
            print(f"  ✅ No empty files")
        
        if result['columns']:
            print(f"  Columns: {result['columns']}")
        
        # Check for null values
        if result['null_counts']:
            null_cols = {k: v for k, v in result['null_counts'].items() if v > 0}
            if null_cols:
                print(f"  ⚠️  Columns with NULL values: {null_cols}")
            else:
                print(f"  ✅ No NULL values found")
        
        if result['errors']:
            print(f"  ❌ Errors: {result['errors']}")
        
        if result['sample_data'] is not None:
            print(f"\n  Sample Data:")
            print(result['sample_data'].to_string().replace('\n', '\n  '))
    
    print("\n" + "="*80)


def main():
    print("🔍 Starting Data Quality Analysis...")
    print(f"Data Directory: {DATA_DIR}")
    
    # Get all category folders
    categories = ['burns', 'mints', 'prices', 'state', 'states', 'swaps']
    
    all_results = []
    for category in categories:
        print(f"\n📂 Analyzing: {category}...")
        folder_path = DATA_DIR / category
        result = analyze_parquet_files(folder_path, category)
        all_results.append(result)
    
    # Print report
    print_analysis_report(all_results)
    
    # Generate visualizations
    print("\n📈 Generating Visualizations...")
    plot_daily_row_counts(all_results)
    plot_file_sizes(all_results)
    plot_data_completeness(all_results)
    plot_missing_dates_heatmap(all_results)
    
    print("\n✅ Analysis Complete! Check the generated PNG files in simulation_6 folder.")
    
    # Summary statistics
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    total_files = sum(r['file_count'] for r in all_results)
    total_rows = sum(r['total_rows'] for r in all_results)
    total_missing = sum(len(r['missing_dates']) for r in all_results)
    total_empty = sum(len(r['empty_files']) for r in all_results)
    
    print(f"  Total Files: {total_files}")
    print(f"  Total Rows: {total_rows:,}")
    print(f"  Total Missing Date Gaps: {total_missing}")
    print(f"  Total Empty Files: {total_empty}")
    
    if total_missing == 0 and total_empty == 0:
        print("\n  🎉 Data appears to be COMPLETE and CONSISTENT!")
    else:
        print("\n  ⚠️  Some data quality issues detected - review above for details")


if __name__ == "__main__":
    main()
