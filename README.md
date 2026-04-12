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
<img width="1344" height="747" alt="image" src="https://github.com/user-attachments/assets/0171f649-4244-4325-899b-70647a5f9887" />


# Results:

<img width="1218" height="439" alt="image" src="https://github.com/user-attachments/assets/98ae3f89-0799-4e78-af98-9cb1d731a0d5" />

Achieved improvement in R² score by up to 10.53%, indicating better prediction accuracy
Increased Signal-to-Noise Ratio (SNR) by up to 26.40%, enhancing output signal quality
Enabled faster and more efficient model training through incremental updates compared to traditional retraining approaches

# Plots:
<img width="1800" height="900" alt="slownfastp_3 6V_45_plot" src="https://github.com/user-attachments/assets/3a25ed3d-972d-4248-bf1e-63fb6d0c182b" />

<img width="1800" height="900" alt="fastnfastp_3 6V_45_plot" src="https://github.com/user-attachments/assets/7d680955-a6cc-46ae-b0c1-d66e465e5fec" />

<img width="1800" height="900" alt="typical_3 3V_45_plot" src="https://github.com/user-attachments/assets/d81b03cc-981f-403b-b7d7-5c23d650a957" />

# Metrics:

<img width="862" height="544" alt="image" src="https://github.com/user-attachments/assets/a8572841-4bd2-4745-882b-bc5f20bc757d" />
