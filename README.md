# Incremental Learning for Analog Mixed Signal Circuit Verification

# Description:
Developed an automated verification framework for Analog Mixed Signal (AMS) circuits using Incremental Learning techniques to eliminate repeated full-model retraining, thereby improving computational efficiency and scalability.

# Key Contributions:
Designed and implemented an incremental machine learning pipeline that updates models dynamically with new data instead of retraining from scratch
Applied advanced regression models including Hoeffding Tree, Aggregated Mondrian Tree, and Extra Trees Regressor for continuous learning on streaming data
Performed waveform analysis and anomaly detection on large-scale AMS datasets to identify deviations and ensure circuit reliability
Automated traditionally manual verification processes, significantly reducing analysis time and human effort

# Tech Stack:
Python, Machine Learning, LSTM, MLP, Visual Studio Code

# Dataset:
Processed 150 CSV files, each with a dimension of 15000 × 34
Covered diverse process corners, voltage levels (3V–3.6V), and temperature variations (5°C–125°C)
Utilized structured input features (vinp, pd, xpd, vdda) to predict output waveform (vinn)

# Results:
Achieved improvement in R² score by up to 10.53%, indicating better prediction accuracy
Increased Signal-to-Noise Ratio (SNR) by up to 26.40%, enhancing output signal quality
Enabled faster and more efficient model training through incremental updates compared to traditional retraining approaches

# Plots:
<img width="1800" height="900" alt="slownfastp_3 6V_45_plot" src="https://github.com/user-attachments/assets/3a25ed3d-972d-4248-bf1e-63fb6d0c182b" />

<img width="1800" height="900" alt="fastnfastp_3 6V_45_plot" src="https://github.com/user-attachments/assets/7d680955-a6cc-46ae-b0c1-d66e465e5fec" />
