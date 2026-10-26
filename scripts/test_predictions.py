"""
test_predictions.py
-------------------
Automated test script verifying predict_ipo() on 3 real unseen IPOs from test.csv.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.predict import predict_ipo

def run_tests():
    test_path = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    test_df = pd.read_csv(test_path)

    # 3 distinct target IPOs covering Apply, Neutral, and Avoid regimes
    test_names = [
        "Navbharat Consumer Care Ltd", # Actual Apply (+15.96%)
        "Synergy Diagnostics Ltd",      # Actual Neutral (+2.89%)
        "Innova Diagnostics Ltd",       # Actual Avoid (-4.29%)
    ]

    print("=" * 75)
    print("VERIFYING PREDICTIONS ON 3 TEST SET IPOS")
    print("=" * 75)

    all_passed = True

    for name in test_names:
        row = test_df[test_df["ipo_name"] == name]
        if row.empty:
            print(f"[FAIL] IPO '{name}' not found in test.csv")
            all_passed = False
            continue

        raw_dict = row.iloc[0].to_dict()
        actual_gain = raw_dict["listing_gain_pct"]
        actual_pos = raw_dict["positive_listing"]
        actual_verdict = raw_dict["verdict"]

        print(f"\nEvaluating: '{name}'")
        print(f"  Sector                : {raw_dict.get('sector')}")
        print(f"  Listing Date          : {raw_dict.get('listing_date')}")
        print(f"  Total Subscription    : {raw_dict.get('total_sub'):.2f}x")
        print(f"  Grey Market Premium   : INR {raw_dict.get('gmp_last_day')}")
        print(f"  Actual Debut Gain     : {actual_gain:+.2f}%")
        print(f"  Actual Verdict        : {actual_verdict}")

        # Run prediction
        res = predict_ipo(raw_dict)

        print("  --- Model Inferences ---")
        print(f"  Market Mood (HMM)     : {res['market_mood']}")
        print(f"  Archetype Cluster     : {res['cluster_id']} - {res['cluster_name']}")
        print(f"  Expected Listing Gain : {res['expected_gain_pct']:+.2f}% (Error: {res['expected_gain_pct'] - actual_gain:+.2f}%)")
        print(f"  Positive Debut Prob   : {res['positive_listing_prob'] * 100:.1f}%")
        print(f"  SVM Verdict           : {res['verdict']}")

        peers = res["similar_ipos"]
        print(f"  Top 5 Peers Retrieved : {len(peers)} peers found")
        for idx, peer in peers.iterrows():
            print(f"    - {peer['ipo_name']} ({peer['sector']}) | Sub: {peer['total_sub']}x | Debut: {peer['listing_gain_pct']:+.2f}% | Sim: {peer['similarity_score']:.4f}")

        # Verification checks
        assert isinstance(res["expected_gain_pct"], float), "Expected gain must be a float"
        assert 0.0 <= res["positive_listing_prob"] <= 1.0, "Positive prob must be in [0, 1]"
        assert res["verdict"] in ["Apply", "Neutral", "Avoid"], f"Invalid verdict: {res['verdict']}"
        assert len(peers) == 5, f"Expected 5 peers, got {len(peers)}"
        assert res["market_mood"] in ["Cold", "Warm", "Hot"], f"Invalid mood: {res['market_mood']}"

    print("\n" + "=" * 75)
    if all_passed:
        print("[SUCCESS] All 3 test IPOs passed validation checks with zero errors.")
    else:
        print("[FAILURE] Some tests encountered issues.")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
