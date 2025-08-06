#!/usr/bin/env python3
"""Generate test data for API benchmarks"""

import pandas as pd
import numpy as np
import os

# Create mock recommendations data
recommendations_data = []

stock_codes = [
    "000001.SZ", "000333.SZ", "000651.SZ", "000725.SZ", "000858.SZ",
    "002236.SZ", "002352.SZ", "002415.SZ", "002714.SZ", "300750.SZ",
    "600000.SH", "600030.SH", "600031.SH", "600036.SH", "600048.SH",
    "600104.SH", "600309.SH", "600438.SH", "600519.SH", "601318.SH"
]

for i, code in enumerate(stock_codes):
    # Generate realistic but mock data
    base_score = np.random.normal(0.6, 0.2)  # Mean score of 0.6
    
    recommendations_data.append({
        'code': code,
        'pe': np.random.normal(15, 5) if np.random.random() > 0.1 else None,
        'pb': np.random.normal(1.5, 0.5) if np.random.random() > 0.1 else None,
        'market_cap': np.random.lognormal(6, 1) if np.random.random() > 0.1 else None,
        'score': max(0, min(1, base_score)),
        'suggest': '✅ BUY' if base_score > 0.7 else 'HOLD',
        'reason': 'Mock data for testing'
    })

# Sort by score
recommendations_data.sort(key=lambda x: x['score'], reverse=True)

# Save to CSV
df = pd.DataFrame(recommendations_data)
df.to_csv('today_recommendations.csv', index=False)

print(f"✅ Generated mock recommendations data with {len(recommendations_data)} stocks")
print("Top 3 recommendations:")
for i in range(3):
    stock = recommendations_data[i]
    print(f"  {stock['code']}: {stock['score']:.3f} ({stock['suggest']})")