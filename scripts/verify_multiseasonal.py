import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from dtcd.utils.seasonality import SeasonalityGenerator

T = 1000
rng = np.random.default_rng(42)
gen = SeasonalityGenerator(random_state=42)

print("=" * 60)
print("MULTI-SEASONAL VERIFICATION")
print("=" * 60)

# Test 1: Single-period seasonality (baseline)
print("\n[Test 1] Single-period (period=24)")
s_single = gen.generate_seasonality(
    n_points=T, periods=[24], strengths=[1.0]
)
print(f"  Shape: {s_single.shape}")
print(f"  Mean: {s_single.mean():.4f}, Std: {s_single.std():.4f}")
print(f"  Min: {s_single.min():.4f}, Max: {s_single.max():.4f}")

# Test 2: Two superimposed periods (daily x weekly)
print("\n[Test 2] Two-period (periods=[24, 168], strengths=[1.0, 0.5])")
s_double = gen.generate_seasonality(
    n_points=T, periods=[24, 168], strengths=[1.0, 0.5]
)
print(f"  Shape: {s_double.shape}")
print(f"  Mean: {s_double.mean():.4f}, Std: {s_double.std():.4f}")

# Test 3: Three superimposed periods
print("\n[Test 3] Three-period (periods=[12, 24, 168], strengths=[0.3, 1.0, 0.5])")
s_triple = gen.generate_seasonality(
    n_points=T, periods=[12, 24, 168], strengths=[0.3, 1.0, 0.5]
)
print(f"  Shape: {s_triple.shape}")
print(f"  Std: {s_triple.std():.4f}")

# Spectral verification: FFT of multi-period signal should show 
# peaks at the specified periods (within numerical tolerance)
print("\n[Spectral check] FFT of two-period signal")
from numpy.fft import rfft, rfftfreq
freqs = rfftfreq(T, d=1.0)
spectrum = np.abs(rfft(s_double))
# Convert top peaks back to periods
top_indices = np.argsort(spectrum)[-5:][::-1]
top_periods = sorted(
    [1.0/freqs[i] if freqs[i] > 0 else float('inf') for i in top_indices],
    reverse=True
)
print(f"  Top 5 periods detected: {[f'{p:.1f}' for p in top_periods]}")
print(f"  Expected periods present in top 5? "
      f"24 in top: {any(abs(p - 24) < 2 for p in top_periods)}, "
      f"168 in top: {any(abs(p - 168) < 10 for p in top_periods)}")

# Test 4: add_seasonality on a DataFrame
print("\n[Test 4] add_seasonality on DataFrame")
import pandas as pd
df = pd.DataFrame({
    'A': rng.normal(size=T),
    'B': rng.normal(size=T),
    'C': rng.normal(size=T),
})
config = {
    'A': {'periods': [24], 'strengths': [1.0]},
    'B': {'periods': [168], 'strengths': [1.0]},
    'C': {'periods': [24, 168], 'strengths': [1.0, 0.5]},
}
df_with_season = gen.add_seasonality(df, config)
print(f"  Original A std: {df['A'].std():.3f}, with seasonality: {df_with_season['A'].std():.3f}")
print(f"  Original B std: {df['B'].std():.3f}, with seasonality: {df_with_season['B'].std():.3f}")
print(f"  Original C std: {df['C'].std():.3f}, with seasonality: {df_with_season['C'].std():.3f}")
print(f"  C should have larger boost (two components): "
      f"{df_with_season['C'].std() > df_with_season['A'].std()}")

print("\n" + "=" * 60)
print("MULTI-SEASONAL: ALL TESTS COMPLETED")
print("=" * 60)
