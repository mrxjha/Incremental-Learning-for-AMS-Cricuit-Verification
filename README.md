# Incremental Learning for Analog Mixed Signal Cricuit Verification

Description:
Developed an automated verification system for Analog Mixed Signal (AMS) circuits using Incremental Learning to reduce retraining time and improve efficiency.

Key Contributions:

Designed an incremental machine learning pipeline to update models without retraining from scratch
Implemented Hoeffding Tree, Mondrian Tree, and Extra Trees Regressors for continuous learning
Performed waveform analysis and anomaly detection on large-scale AMS datasets
Reduced manual verification effort by automating data analysis

Tech Stack:
Python, Machine Learning, LSTM, MLP, VS Code

Dataset:

150 CSV files (15000 × 34 each)
Multiple process corners, voltage levels, and temperature variations

Results:

Improved R² score by up to 10.53%
Increased SNR by up to 26.40%
Achieved faster training with incremental updates
