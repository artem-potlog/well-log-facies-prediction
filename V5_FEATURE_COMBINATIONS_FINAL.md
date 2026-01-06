# V5 Feature Combinations - Final Version

**Date:** 2026-01-02  
**Status:** ✅ Complete with Forward Selection Results  
**Total Combinations:** 19 (18 manual + 1 data-driven)

---

## 🎯 **Overview**

After extensive testing including:
- Phase 3 full progressive (18 manual combinations)
- Forward feature selection (data-driven discovery)
- Hypothesis testing (fluid features, lithology impact)

We identified **19 final feature combinations** organized in 7 groups.

---

## 🏆 **KEY FINDINGS**

### **Best Performers:**
1. **FC_19 (Forward Selection):** 8 features → **0.480** 🥇
2. **FC_13 (FC_06 + RT):** 16 features → **0.392** 🥈
3. **FC_10 (FC_03 + RT):** 8 features → **0.389** 🥉

### **Critical Discoveries:**
- ✅ **VSH is the master feature** (single feature: 0.424!)
- ✅ **RT is valuable** (+3-16% improvement)
- ❌ **RD is toxic** (-15% drop)
- ❌ **SW is problematic** (calculated from noisy RD)
- ✅ **Lithology dominates** (4 of 8 optimal features)
- ✅ **Data-driven beats manual** (+22% improvement)

---

## 📊 **COMPLETE FEATURE COMBINATIONS**

### **GROUP 1: BASELINE PROGRESSION (3 combos)**

#### **FC_01_rock_only (3 features)**
```
Features: NPHI, RHOB, DT
Result: 0.270 (custom_sand)
Purpose: Minimal baseline
```

#### **FC_02_rock_clean (5 features)**
```
Features: GR, VSH, NPHI, RHOB, DT
Result: 0.311 (custom_sand)
Purpose: Add cleanliness indicators
Improvement: +15% over FC_01
```

#### **FC_03_add_perm (7 features)**
```
Features: GR, VSH, NPHI, RHOB, DT, PHIF, KLOGH
Result: 0.334 (custom_sand)
Purpose: Clean baseline with reservoir quality
Improvement: +7% over FC_02
```

---

### **GROUP 2: LITHOLOGY PATH (3 combos)**

#### **FC_04_fc3_lith (12 features)**
```
Features: FC_03 + Lithology_0,1,2,3,4
Result: 0.349 (custom_sand)
Purpose: Test lithology impact on clean baseline
Improvement: +4.5% over FC_03
```

#### **FC_05_fc3_lith_relpos (13 features)**
```
Features: FC_04 + RelPos
Result: 0.372 (custom_sand)
Purpose: Add stratigraphic position
Improvement: +6.6% over FC_04
```

#### **FC_06_fc3_lith_relpos_grtex (15 features)**
```
Features: FC_05 + GR_Slope, GR_Serration
Result: 0.379 (custom_sand)
Purpose: Add texture discrimination
Improvement: +1.9% over FC_05
Note: Clean path winner (no fluid features)
```

---

### **GROUP 3: ENGINEERED FEATURES (3 combos)**

#### **FC_07_fc6_opposition (16 features)**
```
Features: FC_06 + OppositionIndex
Result: 0.383 (custom_sand)
Purpose: Test coarsening vs fining trends
Improvement: +1.1% over FC_06
```

#### **FC_08_fc6_ntg (17 features)**
```
Features: FC_07 + NTG_Slope
Result: 0.383 (custom_sand)
Purpose: Test net-to-gross trends
Note: Stable (std = 0.042)
```

#### **FC_09_fc6_all_eng (18 features)**
```
Features: FC_08 + BaseSharpness
Result: 0.384 (custom_sand)
Purpose: All stratigraphic engineered features (no fluid)
Note: Very stable (std = 0.036) - most consistent model
```

---

### **GROUP 4: FLUID ISOLATION (3 combos)**

#### **FC_10_fc3_rt (8 features)**
```
Features: FC_03 + RT
Result: 0.389 (custom_sand)
Purpose: Test RT impact alone
KEY FINDING: RT adds +16.4% to FC_03! RT is VALUABLE!
```

#### **FC_11_fc3_rt_rd (9 features)**
```
Features: FC_10 + RD
Result: 0.330 (custom_sand)
Purpose: Test RD impact
KEY FINDING: RD causes -15% drop! RD is TOXIC!
```

#### **FC_12_fc3_rt_rd_sw (10 features)**
```
Features: FC_11 + SW
Result: 0.327 (custom_sand)
Purpose: Test SW impact
KEY FINDING: SW adds minimal damage beyond RD
Conclusion: RD is main culprit, SW inherits RD noise
```

---

### **GROUP 5: OPTIMAL + FLUID (3 combos)**

#### **FC_13_fc6_rt (16 features)**
```
Features: FC_06 + RT
Result: 0.392 (custom_sand)
Purpose: Add RT to optimal clean model
Improvement: +3.4% over FC_06
Note: Manual design winner (before forward selection)
```

#### **FC_14_fc6_rt_rd (17 features)**
```
Features: FC_13 + RD
Result: 0.354 (custom_sand)
Purpose: Test if RD hurts strong foundation
Finding: -9.7% drop - RD toxic even with strong base
```

#### **FC_15_fc6_all_fluid (18 features)**
```
Features: FC_14 + SW
Result: 0.361 (custom_sand)
Purpose: All features except truly bad ones
Note: SW slightly helps when combined with everything else
```

---

### **GROUP 6: FORWARD SELECTION WINNER (1 combo)** ⭐

#### **FC_19_forward_optimal (8 features)** 🏆
```
Features: VSH, RelPos,
          Lithology_3, Lithology_0, Lithology_1, Lithology_4,
          OppositionIndex, NTG_Slope

Result: 0.480 (custom_sand) - BEST SCORE ACHIEVED
Stability: 0.060 (std) - Good stability
Purpose: DATA-DRIVEN OPTIMAL - Forward selection discovery

KEY INSIGHT:
- Only 1 traditional log (VSH)
- No rock physics triad (NPHI, RHOB, DT)
- No porosity/permeability (PHIF, KLOGH)
- No resistivity (RT, RD)
- Focus: Cleanliness + Lithology + Stratigraphic context

Why it wins:
1. VSH captures energy level (primary tidal control)
2. 4 lithology columns capture rock type
3. RelPos + trends capture stratigraphic architecture
4. No redundant/noisy features

Samples per feature: 466 (optimal ratio)
```

---

### **GROUP 7: SPECIAL CASES (3 combos)**

#### **FC_16_measured_only (11 features)**
```
Features: GR, RT, NPHI, RHOB, RD, DT, RelPos, GR_Slope, GR_Serration, 
          OppositionIndex, BaseSharpness
Result: 0.273 (custom_sand)
Purpose: For F-14 well (no PHIF/SW/KLOGH/VSH)
Note: Poor performance - missing key features
```

#### **FC_17_max_features (21 features)**
```
Features: All 21 features
Result: 0.346 (custom_sand)
Purpose: Test everything together
Note: Overfitting - too many features for data size
```

#### **FC_18_tidal_optimized (13 features)**
```
Features: Domain expert curated mix
Result: 0.355 (custom_sand)
Purpose: Domain knowledge vs data-driven
Note: Data-driven (FC_19) beats expert by +35%!
```

---

## 📈 **PERFORMANCE COMPARISON**

### **By Approach:**

| Approach | Best Model | Features | Score | Method |
|----------|------------|----------|-------|--------|
| **Data-Driven** | FC_19 | 8 | **0.480** | Forward selection |
| Manual - Strong | FC_13 | 16 | 0.392 | Domain + testing |
| Manual - Clean | FC_06 | 15 | 0.379 | Progressive design |
| Manual - Simple | FC_10 | 8 | 0.389 | Add RT to FC_03 |
| Domain Expert | FC_18 | 13 | 0.355 | Expert curation |

**Winner:** Data-driven forward selection! 🎯

---

### **By Feature Count:**

| Count | Best Score | Model | Insight |
|-------|------------|-------|---------|
| 3 | 0.270 | FC_01 | Too simple |
| 5 | 0.311 | FC_02 | Still simple |
| 7 | 0.334 | FC_03 | Baseline |
| **8** | **0.480** | **FC_19** | **OPTIMAL!** ⭐ |
| 13 | 0.372 | FC_05 | Good |
| 15 | 0.379 | FC_06 | Very good |
| 16 | 0.392 | FC_13 | Manual best |
| 18 | 0.384 | FC_09 | Overfitting starts |
| 21 | 0.346 | FC_17 | Too complex |

**Optimal range: 8-16 features**

---

## 🔬 **FEATURE IMPORTANCE RANKING**

**From forward selection order and results:**

### **Tier 1: Critical (Must Have)**
1. **VSH** - Master feature (single: 0.424)
2. **Lithology columns** - 4 of top 8 features
3. **RelPos** - Stratigraphic position (added 2nd)

### **Tier 2: Important**
4. **OppositionIndex** - Coarsening/fining trends
5. **NTG_Slope** - Net-to-gross trends
6. **RT** - Resistivity (when used alone or with strong base)

### **Tier 3: Marginal**
7. **GR** - Redundant with VSH
8. **GR_Slope, GR_Serration** - Texture features
9. **BaseSharpness** - Contact features

### **Tier 4: Not Needed**
10. **NPHI, RHOB, DT** - Rock physics (redundant with lithology)
11. **PHIF, KLOGH** - Reservoir properties (redundant)
12. **GR** - Captured by VSH

### **Tier 5: Toxic**
13. **RD** - Deep resistivity (causes -15% drop!)
14. **SW** - Water saturation (inherits RD noise)

---

## 💡 **KEY INSIGHTS**

### **1. Energy > Everything**
VSH (energy proxy) alone beats most multi-feature models!
**In tidal systems:** Energy level is THE primary facies control

### **2. Lithology is Essential**
4 of 8 optimal features are lithology
**Without lithology:** Can't distinguish facies properly

### **3. Stratigraphic Context Matters**
RelPos + OppositionIndex + NTG_Slope = stratigraphic architecture
**These capture:** Vertical position + trends + sequences

### **4. Traditional Features Overrated**
NPHI, RHOB, DT not selected by forward selection
**Why?** Lithology already captures rock physics information

### **5. Resistivity is Complex**
- RT alone: Excellent (+16%)
- RT + RD: Terrible (-15%)
**Lesson:** Feature quality matters more than quantity

---

## 🎯 **RECOMMENDATIONS FOR PRODUCTION**

### **Best Single Model:**
**Use FC_19 (forward selection optimal)**
- 8 features: VSH, RelPos, Lith_0,1,3,4, OppositionIndex, NTG_Slope
- Score: 0.480 ± 0.060
- Simple, focused, data-driven

### **Alternative (If you want more features):**
**Use FC_13 (manual winner)**
- 16 features: FC_06 + RT
- Score: 0.392 ± 0.082
- More features, but lower performance

### **Ensemble Strategy:**
**Use top 10 models from 57-model run:**
- Expected accuracy: ~49-51%
- Best uncertainty quantification
- Combines manual and data-driven approaches

---

## 📚 **TESTING HISTORY**

### **Phase 1-3: Manual Progressive (30 models)**
- Discovered FC_03 (7 feat) clean baseline
- Discovered FC_06 (15 feat) clean path winner
- Identified RT vs RD problem

### **Hypothesis Test: FC_03 Variations**
- Confirmed lithology helps (+1.2%)
- Discovered need for synergistic features
- Led to FC_06 as manual winner

### **Full Progressive: 54 Models**
- Tested all 18 manual combinations
- Discovered RT is valuable, RD is toxic
- FC_13 emerged as manual best (0.392)

### **Forward Selection: 3 Hours**
- Systematic single-feature testing
- Progressive build: 1→2→3→...→8 features
- Discovered FC_19 optimal (0.480)
- **+22% better than manual design!**

---

## ✅ **FINAL FEATURE COMBINATION LIST**

**Total: 19 combinations**

**Progressive Testing (FC_01-18):**
- Baseline: FC_01-03
- Lithology: FC_04-06
- Engineered: FC_07-09
- Fluid isolation: FC_10-12
- Optimal + fluid: FC_13-15
- Special: FC_16-18

**Data-Driven (FC_19):**
- Forward selection optimal ⭐

**Total models: 19 × 3 metrics = 57 models**

---

## 🎓 **THESIS IMPLICATIONS**

### **Novel Contributions:**

**1. Feature Quality Assessment**
- Identified valuable (RT, VSH, lithology)
- Identified toxic (RD, SW)
- Quantified impact of each

**2. Feature Selection Methodology**
- Manual design: 39.2%
- Data-driven: 48.0%
- **Proved data-driven superiority**

**3. Tidal-Specific Insights**
- Energy level (VSH) is primary control
- Lithology essential (4 of 8 features)
- Stratigraphic trends matter more than texture

**4. Sample Size Guidelines**
- Optimal: 8 features for 3730 samples
- Ratio: ~466 samples/feature
- Guideline: 400-500 samples/feature for Random Forest

---

**FEATURE COMBINATIONS FINALIZED ✅**

*Ready for production use with 19 systematic combinations!*
