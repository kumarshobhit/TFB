self.head_dim = d_model // num_heads  # e.g., 128 / 8 = 16

# This creates MULTIPLE theta values (one per dimension pair)
theta = 1.0 / (base_freq ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
```

**For your config (d_model=128, n_heads=8):**
- `head_dim = 16`
- You have **8 different theta values** (for dimension pairs 0-1, 2-3, 4-5, ..., 14-15)

**With base_freq=500:**
```
theta_0 = 1.0 / (500^0.000) = 1.0000       → Fastest rotation
theta_1 = 1.0 / (500^0.125) ≈ 0.7197       
theta_2 = 1.0 / (500^0.250) ≈ 0.5180
...
theta_7 = 1.0 / (500^0.875) ≈ 0.0268       → Slowest rotation
```

### **2. The Rotation Formula (What She's Explaining)**

For positions m and n at dimension i:
```
Rotation angle: φ = (m - n) × theta_i
```

**Example:** Comparing position 10 and position 34:
- Relative distance: m - n = -24
- For dimension pair 0: φ₀ = -24 × 1.0000 = -24.0 radians
- For dimension pair 7: φ₇ = -24 × 0.0268 = -0.643 radians

**Different dimensions rotate at different speeds!**

### **3. The Frequency Spectrum**

The **effective frequency** for each dimension is:
```
f_i = theta_i / (2π)
```

**For base_freq=500, your RoPE has frequencies:**
```
f_0 ≈ 0.159 Hz  (wavelength ≈ 6.28 positions)   ← Fast, captures local patterns
f_1 ≈ 0.115 Hz  (wavelength ≈ 8.73 positions)
f_2 ≈ 0.082 Hz  (wavelength ≈ 12.1 positions)
f_3 ≈ 0.059 Hz  (wavelength ≈ 16.9 positions)
f_4 ≈ 0.043 Hz  (wavelength ≈ 23.5 positions)   ← Around your period!
f_5 ≈ 0.031 Hz  (wavelength ≈ 32.7 positions)
f_6 ≈ 0.022 Hz  (wavelength ≈ 45.5 positions)
f_7 ≈ 0.004 Hz  (wavelength ≈ 234 positions)    ← Slow, captures long-range