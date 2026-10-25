"""
make_sample_data.py
-------------------
Generates a clearly labeled SYNTHETIC dataset of 300 Indian IPOs for pipeline
testing and verification before real historical data is ready.

⚠️ WARNING: ALL RECORDS IN THIS DATASET ARE SYNTHETIC AND ALGORITHMICALLY GENERATED.
DO NOT USE THIS DATA FOR REAL-WORLD FINANCIAL TRADING OR INVESTMENT DECISIONS.
"""

import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Loud console banner (ASCII-compatible for Windows console)
SYNTHETIC_WARNING_BANNER = """
================================================================================
[CRITICAL WARNING] SYNTHETIC DATASET GENERATION
The data produced by this script is purely SYNTHETIC for educational and
pipeline testing purposes. It does NOT represent actual audited market data.
================================================================================
"""

SECTORS = [
    "Technology",
    "Financial Services",
    "Healthcare",
    "Consumer Goods",
    "Manufacturing",
    "Automobile",
    "Infrastructure",
    "Chemicals",
    "Energy",
    "Retail",
]

COMPANY_PREFIXES = [
    "Apex", "Zenith", "Tata", "Bharat", "Navbharat", "Reliance", "Kalyan",
    "Synergy", "Omni", "Vanguard", "Indus", "Pragati", "Shree", "Auro",
    "Kaveri", "Sterling", "Paramount", "Horizon", "Innova", "Maximus",
    "Quantum", "Adani", "Matrix", "Pulse", "Finova", "Credence", "Aegis",
    "Vistara", "Solaris", "Integra"
]

COMPANY_SUFFIXES = [
    "Technologies Ltd", "Fintech Solutions Ltd", "Infra Projects Ltd",
    "Diagnostics Ltd", "Chemicals Ltd", "Enterprises Ltd", "Consumer Care Ltd",
    "Engineering Ltd", "Mobility Systems Ltd", "Pharma Ltd", "Logistics Ltd",
    "Renewable Power Ltd", "Agro Industries Ltd", "Capital Services Ltd",
    "Digital Networks Ltd"
]


def generate_synthetic_ipo_dataset(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic dataset of Indian IPOs with intentional missing values.

    Parameters
    ----------
    n_samples : int, default=300
        Number of synthetic IPO records to generate.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame matching the schema expected by src/data_loader.py.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate unique IPO names
    names = []
    used_names = set()
    while len(names) < n_samples:
        prefix = rng.choice(COMPANY_PREFIXES)
        suffix = rng.choice(COMPANY_SUFFIXES)
        name = f"{prefix} {suffix}"
        if name not in used_names:
            used_names.add(name)
            names.append(name)

    # 2. Generate chronological listing dates between Jan 2021 and Dec 2024
    start_date = datetime(2021, 1, 15)
    end_date = datetime(2024, 12, 15)
    total_days = (end_date - start_date).days
    random_day_offsets = np.sort(rng.integers(0, total_days, size=n_samples))
    listing_dates = [(start_date + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in random_day_offsets]

    # 3. Categoricals and Issue Specs
    sectors = rng.choice(SECTORS, size=n_samples, p=[0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04])
    
    # Issue size in Crores (log-normal distribution, typical Indian mainboard IPOs: 200 Cr to 15,000 Cr)
    issue_size_cr = np.round(np.exp(rng.normal(loc=6.8, scale=0.8, size=n_samples)), 2)
    issue_size_cr = np.clip(issue_size_cr, 120.0, 22000.0)

    # Price band upper in INR (typical Rs. 70 to Rs. 2,800)
    price_band_upper = np.round(np.exp(rng.normal(loc=5.8, scale=0.7, size=n_samples)), 2)
    price_band_upper = np.clip(price_band_upper, 45.0, 3200.0)

    # Issue price is almost always set at the upper cut-off in Indian book-building
    issue_price = price_band_upper.copy()

    # 4. Subscription numbers (multiples: QIB, NII, Retail)
    # Retail subscription usually 1x to 70x
    retail_sub = np.round(rng.gamma(shape=2.5, scale=6.0, size=n_samples), 2)
    retail_sub = np.clip(retail_sub, 0.3, 140.0)

    # NII (Non-institutional / HNI) often swings wide: 0.5x to 350x
    nii_sub = np.round(rng.gamma(shape=1.8, scale=18.0, size=n_samples), 2)
    nii_sub = np.clip(nii_sub, 0.2, 450.0)

    # QIB (Qualified Institutional Buyers) swings wide on popular issues
    qib_sub = np.round(rng.gamma(shape=2.0, scale=20.0, size=n_samples), 2)
    qib_sub = np.clip(qib_sub, 0.1, 380.0)

    # Total subscription (weighted approx: 50% QIB, 15% NII, 35% Retail)
    total_sub = np.round(0.50 * qib_sub + 0.15 * nii_sub + 0.35 * retail_sub, 2)

    # 5. Grey Market Premium (last day before listing in INR)
    # GMP is roughly correlated with subscription demand and general market sentiment
    expected_gmp_pct = 0.45 * np.log1p(total_sub) * 12 + rng.normal(0, 15, size=n_samples)
    gmp_last_day = np.round(issue_price * (expected_gmp_pct / 100.0), 2)
    # Can be slightly negative for weak issues
    gmp_last_day = np.clip(gmp_last_day, -0.20 * issue_price, 1.8 * issue_price)

    # Realistic missing values in GMP (~15% of IPOs have unquoted or missing GMP data)
    missing_gmp_idx = rng.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    gmp_last_day[missing_gmp_idx] = np.nan

    # 6. Issue Breakdown (Fresh Issue % vs OFS %)
    fresh_issue_pct = np.round(rng.uniform(10.0, 100.0, size=n_samples), 2)
    ofs_pct = np.round(100.0 - fresh_issue_pct, 2)

    # 7. Financial Fundamentals (with intentional realistic missing values)
    # Revenue growth YoY % (-10% to 150%)
    revenue_growth_pct = np.round(rng.normal(loc=22.0, scale=25.0, size=n_samples), 2)
    revenue_growth_pct = np.clip(revenue_growth_pct, -35.0, 220.0)
    missing_rev_idx = rng.choice(n_samples, size=int(0.08 * n_samples), replace=False)
    revenue_growth_pct[missing_rev_idx] = np.nan

    # Profit growth YoY % (-30% to 200%)
    profit_growth_pct = np.round(revenue_growth_pct * 0.8 + rng.normal(loc=5.0, scale=30.0, size=n_samples), 2)
    missing_profit_idx = rng.choice(n_samples, size=int(0.12 * n_samples), replace=False)
    profit_growth_pct[missing_profit_idx] = np.nan

    # P/E vs Peers ratio (0.5 to 2.8, where 1.0 means fairly priced vs sector peers)
    pe_vs_peers = np.round(rng.lognormal(mean=0.05, sigma=0.35, size=n_samples), 2)
    pe_vs_peers = np.clip(pe_vs_peers, 0.4, 3.5)
    missing_pe_idx = rng.choice(n_samples, size=int(0.10 * n_samples), replace=False)
    pe_vs_peers[missing_pe_idx] = np.nan

    # 8. Macro Market Condition (Nifty 50 7-day trend % prior to listing)
    nifty_7d_trend_pct = np.round(rng.normal(loc=0.3, scale=1.8, size=n_samples), 2)
    nifty_7d_trend_pct = np.clip(nifty_7d_trend_pct, -6.5, 6.5)

    # 9. Realistic Listing Price calculation
    # Driven primarily by GMP, Total Subscription, Nifty Trend, and Valuation, with random market noise
    gmp_contribution = np.where(np.isnan(gmp_last_day), 0.0, (gmp_last_day / issue_price) * 100.0)
    sub_contribution = np.log1p(total_sub) * 7.5
    macro_contribution = nifty_7d_trend_pct * 2.5
    valuation_penalty = np.where(np.isnan(pe_vs_peers), 0.0, (pe_vs_peers - 1.0) * -6.0)
    noise = rng.normal(0, 8.0, size=n_samples)

    # Blended listing gain %
    synthetic_listing_gain_pct = (
        0.50 * gmp_contribution +
        0.30 * sub_contribution +
        0.10 * macro_contribution +
        0.10 * valuation_penalty +
        noise
    )
    # Clip realistic listing gains between -40% (deep discount) to +220% (blockbuster debut)
    synthetic_listing_gain_pct = np.clip(synthetic_listing_gain_pct, -40.0, 220.0)
    
    # Calculate listing price from gain percentage
    listing_price = np.round(issue_price * (1.0 + synthetic_listing_gain_pct / 100.0), 2)

    df = pd.DataFrame({
        "ipo_name": names,
        "listing_date": listing_dates,
        "sector": sectors,
        "issue_size_cr": issue_size_cr,
        "price_band_upper": price_band_upper,
        "issue_price": issue_price,
        "listing_price": listing_price,
        "qib_sub": qib_sub,
        "nii_sub": nii_sub,
        "retail_sub": retail_sub,
        "total_sub": total_sub,
        "gmp_last_day": gmp_last_day,
        "fresh_issue_pct": fresh_issue_pct,
        "ofs_pct": ofs_pct,
        "revenue_growth_pct": revenue_growth_pct,
        "profit_growth_pct": profit_growth_pct,
        "pe_vs_peers": pe_vs_peers,
        "nifty_7d_trend_pct": nifty_7d_trend_pct,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Indian IPO dataset for testing.")
    parser.add_argument("--samples", type=int, default=300, help="Number of IPO records to generate (default: 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("data", "raw"),
        help="Directory to store generated data files",
    )
    args = parser.parse_args()

    print(SYNTHETIC_WARNING_BANNER)

    os.makedirs(args.output_dir, exist_ok=True)

    df = generate_synthetic_ipo_dataset(n_samples=args.samples, seed=args.seed)

    # 1. Primary loudly-named synthetic file
    warning_filename = "synthetic_ipo_data_WARNING_SYNTHETIC.csv"
    warning_path = os.path.join(args.output_dir, warning_filename)
    df.to_csv(warning_path, index=False)
    print(f"[OK] Generated {len(df)} synthetic records -> {warning_path}")

    # 2. Also populate standard raw file ipo_data.csv if it does not exist
    default_path = os.path.join(args.output_dir, "ipo_data.csv")
    if not os.path.exists(default_path):
        df.to_csv(default_path, index=False)
        print(f"[INFO] Created fallback mock file for immediate pipeline testing -> {default_path}")
    else:
        print(f"[INFO] Existing file '{default_path}' preserved (not overwritten).")


if __name__ == "__main__":
    main()
