from flask import Flask, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__) # สร้างตัวแปร app ให้ Render เจอ

@app.route('/')
def home():
    # --- ส่วนโค้ดคำนวณเดิมของคุณ ---
    FILE_NAME = 'Data for econometrics.xlsx'
    X_COLS = ['X1 : Ad Expense Thousand THB', 'X2 : GDP (Billion THB)', 'X3 : CPI']
    Y_COL = 'Y : Revennue Thousand THB'

    try:
        df = pd.read_excel(FILE_NAME)
        df.columns = df.columns.str.strip()
        df_clean = df[X_COLS + [Y_COL]].dropna()

        X = df_clean[X_COLS]
        Y = df_clean[Y_COL]
        X_with_const = sm.add_constant(X)
        model = sm.OLS(Y, X_with_const).fit()

        # สร้างกราฟ
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        for i, col_name in enumerate(X_COLS):
            X_sub = df_clean[[col_name]].values
            Y_sub = df_clean[Y_COL].values
            line_model = LinearRegression().fit(X_sub, Y_sub)
            Y_pred = line_model.predict(X_sub)
            axes[i].scatter(X_sub, Y_sub, alpha=0.5)
            axes[i].plot(X_sub, Y_pred, color='black', linestyle='--')
            axes[i].set_title(f'Impact of {col_name}')

        plt.tight_layout()
        plt.savefig('output_plot.png') # เซฟกราฟชั่วคราว
        
        # ส่งรูปกราฟออกไปโชว์ที่หน้าเว็บ
        return send_file('output_plot.png', mimetype='image/png')

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
