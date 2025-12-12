import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Methodology for Blood Pressure Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("Methodology for Blood Pressure Prediction from Consumer Wearables")
st.markdown("*Technical methodology for cuffless BP monitoring using Garmin HRV data*")

st.header("Executive Summary")
st.markdown("""
This application implements a two-stage approach for non-invasive blood pressure monitoring using consumer wearable devices:
1. *Deep Learning Prediction Model*: A ResNet152-based neural network trained on the MIMIC-BP dataset to predict BP from cardiac signals.
2. *Personal Calibration System*: Individual-specific calibration using reference cuff measurements to improve accuracy and overcome bias from the deep learning model dataset.
""")

st.header("Deep Learning BP Prediction Model")

st.subheader("Dataset: MIMIC-BP")
st.markdown("""
**Source**: [Harvard Dataset - MIMIC-BP: A curated dataset for blood pressure estimation](https://www.nature.com/articles/s41597-024-04041-1)

**Dataset Characteristics**:
- **Size**: 1,524 ICU patients from Beth Israel Deaconess Medical Center
- **Duration**: 2008-2019
- **Signals**: Photoplethysmography (PPG), Electrocardiography (ECG), Arterial Blood Pressure (ABP), Respiratory signals
- **Sampling Rate**: 125 Hz
- **Technology Used**: Invasive arterial BP measurements (gold standard)
- **Total Segments**: 27,513 clean waveform segments (30 seconds each)

**Why MIMIC-BP?**
- Clinical-grade data (invasive ABP monitoring)
- Multi-modal physiological signals
- Diverse patient population with various cardiovascular conditions
- Standardized preprocessing and quality control
""")

st.subheader("Model Architecture: ResNet152")
st.markdown("""
**Architecture Selection**:
- Deep Residual Network with 152 layers adapted for 1D time-series signals
- Originally designed for image classification [He et al., 2016](https://arxiv.org/abs/1512.03385)
- Skip connections prevent vanishing gradient problem in very deep networks
- This architecture is recommended in the MIMIC-BP paper as it provided the best performance among various architectures tested.

**Detailed Architecture**:

**Initial Convolution Block**:
- Conv1D: 3 -> 64 channels, kernel=7, stride=2, padding=3
- BatchNorm1D
- ReLU activation
- MaxPool1D: kernel=3, stride=2, padding=1

**Residual Layers** (bottleneck blocks with skip connections):
- **Layer 1**: 64 channels, 3 residual blocks, stride=1
- **Layer 2**: 128 channels, 8 residual blocks, stride=2
- **Layer 3**: 256 channels, 36 residual blocks, stride=2
- **Layer 4**: 512 channels, 3 residual blocks, stride=2

*Total residual blocks: 3 + 8 + 36 + 3 = 50 blocks*

**Residual Block Structure** (each block):
```
Conv1D(3×3) -> BatchNorm -> ReLU -> Dropout(0.2)
-> Conv1D(3×3) -> BatchNorm -> Add(skip connection) -> ReLU
```

**Output Head**:
- Adaptive Average Pooling (global pooling)
- Flatten
- Dropout(0.2)
- Fully Connected Layer: 512 -> 2 (SBP, DBP)

**Total Parameters**: approximately 60 million trainable parameters

**Input Processing**:
- **Window Size**: 10 seconds (1,250 samples at 125 Hz)
- **Channels**: 3 synchronized physiological signals
  - Channel 0: Electrocardiography (ECG)
  - Channel 1: Photoplethysmography (PPG)
  - Channel 2: Respiratory signal
- **Input Shape**: (Batch, 3 channels, 1250 timesteps)

**Why 10-second windows?**
- Captures multiple cardiac cycles (typically 8-15 cycles depending on heart rate)
- Balance between temporal context and computational efficiency
- Aligns with clinical BP measurement protocols

**Output**:
- Direct regression to blood pressure values
- SBP (Systolic Blood Pressure) in mmHg
- DBP (Diastolic Blood Pressure) in mmHg
- No classification (continuous value prediction)
""")

st.subheader("Training Procedure")
st.markdown("""
**Training Dataset Split** (Patient-wise):
- Training: 1,067 patients (~70%)
- Validation: 229 patients (~15%)
- Test: 228 patients (~15%)
- **Critical**: Patient-wise split ensures no data leakage between sets as recommended in the MIMIC-BP paper

**Data Augmentation** (Training only):
- Overlapping windows with 50% stride (5-second overlap)
- Increases training samples without mixing patients between sets
- Validation/Test use non-overlapping windows only

**Training Configuration**:
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
  - Learning Rate: 0.001 (initial)
  - Weight Decay: 1e-4 (L2 regularization)
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Reduces LR by 0.5x when validation loss plateaus
  - Patience: 5 epochs
- **Batch Size**: 64
- **Epochs**: 100 (with early stopping)
- **Early Stopping**: Patience of 15 epochs on validation loss
- **Regularization**:
  - Dropout: 0.2 in residual blocks and before final FC layer
  - Weight decay: 1e-4
  - BatchNorm in all layers

**Hardware**:
- Trained on Apple Silicon (MPS) but code can be run on CUDA GPU
- Training time: I took 2.83 hours on Apple Silicon (MPS) with early stopping on epoch 17.

**Performance Metrics**:
- **Test Set Results**:
  - SBP MAE: ~13 mmHg
  - DBP MAE: ~9 mmHg
  - SBP RMSE: ~16 mmHg
  - DBP RMSE: ~11 mmHg
  - Pearson correlation: r > 0.7 for both SBP and DBP

**Model Selection**:
- Best model selected based on validation loss
- Final evaluation on held-out test set (228 patients)
- Checkpoint saved as `best_model_full.pth`
""")

st.subheader("Test Set Results Visualizations")

st.markdown("**Error Distribution Analysis:**")
try:
    st.image("model/test/error_analysis.png",
             caption="Error distribution plots showing prediction errors for SBP and DBP on test set. "
                     "Includes histograms, scatter plots of predicted vs. actual values, and residual plots.",
             use_container_width=True)
except:
    st.warning("Error analysis image not found at model/test/error_analysis.png")

# Bland-Altman Plots
st.markdown("**Bland-Altman Agreement Analysis:**")
try:
    st.image("model/test/bland_altman_plots.png",
             caption="Bland-Altman plots for SBP and DBP showing agreement between predicted and reference values. "
                     "Dashed lines indicate mean bias and 95% limits of agreement (±1.96 SD).",
             use_container_width=True)
except:
    st.warning("Bland-Altman plots not found at model/test/bland_altman_plots.png")

st.subheader("Adaptation to Garmin Wearables")
st.markdown("""
**Challenge**: The model was trained on clinical ICU signals, but we're using consumer wearable data

**The Garmin Wearable (proof of concept for future applications with other consumer wearables)**:
- Provides: Beat-to-beat R-R intervals (heart rate variability data)
- Does NOT provide: Direct PPG, ECG, or respiratory waveforms

**Signal Reconstruction**:
1. **Heart Rate Signal**: Convert R-R intervals -> instantaneous heart rate -> resampled to 125 Hz
2. **Pseudo-PPG**: Synthesized from heart rate variability using cubic spline interpolation
3. **Pseudo-ECG**: Reconstructed QRS complexes based on R-R interval timing
4. **Pseudo-Respiratory**: Estimated from heart rate variability using respiratory sinus arrhythmia

**Limitations**:
- Reconstructed signals lack fine-grained morphological features
- Higher uncertainty compared to direct sensor measurements
- **This is why personal calibration is critical**
""")

st.image("garmin/figures/garmin_hrm_processing_pipeline.png",
         caption="Example of    Garmin signal processing pipeline."
         "Device heart rate variability data processed into 10-second windows for deep learning input.",
         use_container_width=True)

st.divider()

st.header("Personal Calibration System")
st.markdown("""
**Why is calibration necessary?**

1. **Domain Gap**: Model trained on ICU patients with invasive monitoring but applied to healthy individuals with consumer wearables. Furthermore, the paper notes that "It can be seen, mainly on the DBP histogram, that the values are lower than expected for the healthier population (because the original MIMIC database was collected on patients at intensive care units). Nevertheless, the histogram  variability conforms to the required by ISO 81060-2:201823 if the histograms are shifted to the right by specific amounts. If 21.2 mmHg is added to the DBP values and 11.8 mmHg is added to SBP, the blood pressure distribution nearly attends the required by the ISO standard..."
2. **Signal Quality Difference**: Clinical sensors vs. reconstructed signals from HRV data
3. **Individual Variability**: BP measurement is inherently person-specific due to:
   - Arterial stiffness variations
   - Cardiac output differences
   - Peripheral resistance
   - Body composition

**Scientific Basis**:
- Personal calibration is standard practice in cuffless BP research
- Reduces systematic bias while preserving individual BP variations
- Addressed in MIMIC-BP paper as necessary step for real-world deployment
""")

st.subheader("Calibration Methodology")
st.markdown("""
**Calibration Protocol**:

**Requirements**:
- Minimum 3 paired measurements (Garmin recording + reference cuff BP)
- Measurements should span different BP ranges (rest, post-exercise, different times of day)

**Mathematical Approach**:

**Linear Calibration** (Default):
```
BP_calibrated = α x BP_predicted + β
```
- Fits linear transformation to correct systematic bias
- Requires minimum 3 calibration points
- Assumes consistent offset/scaling error

**Polynomial Calibration**:
```
BP_calibrated = β₀ + β₁xBP + β₂xBP² + ... + βₙxBPⁿ
```
- Handles non-linear relationships
- Requires minimum 5 calibration points
- Degree 2-3 recommended to avoid overfitting

**Hybrid Calibration**:
- The calibration can use different models for SBP and DBP independently
- The best results were Polynomial for SBP and Linear for DBP.
""")

st.subheader("Validation Metrics")
st.markdown("""

**Mean Absolute Error (MAE)**:
- Average absolute difference between calibrated predictions and cuff measurements
- Target: < 10 mmHg for clinical utility

**Standard Deviation (SD)**:
- Variability of errors
- Target: < 8 mmHg for consistency

**Clinical Standards - IEEE 1708a-2019**:
- **Grade A**: MAE ≤ 5 mmHg, SD ≤ 8 mmHg
- **Grade B**: MAE ≤ 10 mmHg, SD ≤ 12 mmHg
- **Grade C**: MAE ≤ 15 mmHg, SD ≤ 15 mmHg

**Statistical Significance**:
- **Paired t-test**: Tests if calibration improvement is statistically significant
- **Cohen's d**: Measures effect size of improvement
  - Large: |d| > 0.8
  - Medium: |d| > 0.5
  - Small: |d| > 0.2

**Bland-Altman Analysis**:
- Plots difference vs. mean for calibrated predictions vs. cuff BP
- Shows bias and limits of agreement (±1.96 SD)
- Standard method for agreement between measurement methods
""")

st.divider()

st.header("Data Flow")

st.markdown("""
**1. Data Acquisition (Garmin Device)**
```
Garmin -> R-R Intervals (HRV) -> .FIT File
```

**2. Signal Processing**
```
.FIT File -> Extract R-R intervals -> Multi-window segmentation -> 10-second windows -> Signal reconstruction (3 channels)
```

**3. Deep Learning Prediction**
```
3-channel signals -> ResNet152 Model -> Raw BP predictions (SBP, DBP)
```

**4. Personal Calibration** (if available)
```
Raw BP + Cuff measurements -> Calibration model -> Calibrated BP
```

**5. Statistical Analysis**
```
Multiple windows -> Mean ± SD -> Temporal trends -> Confidence intervals
```

**6. Clinical Reporting**
```
Calibrated BP -> BP Category -> Risk Assessment -> User Dashboard
```
""")

st.divider()

# Limitations
st.header("Limitations and Future Work")

st.markdown("""
**Current Limitations**:

1. **Signal Reconstruction**: HRV-derived signals lack morphological details present in direct PPG/ECG
2. **Calibration Requirements**: Requires access to reference cuff measurements
3. **Activity State**: Model trained on resting ICU data; may not generalize to active states
4. **Population Bias**: MIMIC-BP is ICU patients; may not represent healthy population
5. **Temporal Validity**: Calibration may drift over time (similar to commercial devices)
""")

st.header("Clinical Context and Applications")

st.markdown("""
**Intended Use**:
- **Research and educational purposes**
- Personal health tracking and trends
- Identification of potential BP abnormalities for clinical follow-up
- **NOT a substitute for clinical diagnosis or treatment decisions**
""")

st.divider()

# Acknowledgments
st.header("Acknowledgments")
st.markdown("""
This project leverages the MIMIC-BP dataset published by Li et al. (2024) in Nature Scientific Data.
We acknowledge the contribution of the original authors in creating and sharing this valuable resource
for cuffless blood pressure monitoring research.

The ResNet architecture is adapted from He et al. (2016) and implemented using PyTorch.

Personal calibration methodology follows clinical best practices established by commercial
cuffless BP devices and IEEE standardization efforts.
""")
