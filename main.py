from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import io
import base64
import os

app = Flask(__name__)

# ตั้งค่าให้ Matplotlib ไม่ต้องเปิดหน้าต่างกราฟ (ป้องกัน Error บน Server)
plt.switch_backend('Agg')

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    correlation_html = None
    regression_summary = None
    prediction = None
    
    # ดึงรายชื่อคอลัมน์จากไฟล์ที่อัปโหลด (ถ้ามี)
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                df = pd.read_excel(file)
                df.columns = df.columns.str.strip()
                
                # ตัวแปรคงที่ตามไฟล์ของคุณ
                X_COLS = ['X1 : Ad Expense Thousand THB', 'X2 : GDP (Billion THB)', 'X3 : CPI']
                Y_COL = 'Y : Revennue Thousand THB'
                
                # ตรวจสอบว่ามีคอลัมน์ครบไหม
                available_cols = [c for c in X_COLS + [Y_COL] if c in df.columns]
                df_clean = df[available_cols].dropna()

                # 1. การคำนวณ Correlation
                corr_matrix = df_clean.corr()
                correlation_html = corr_matrix.to_html(classes='table table-hover table-bordered text-center')

                # 2. การคำนวณ Regression (OLS)
                X = sm.add_constant(df_clean[X_COLS])
                y = df_clean[Y_COL]
                model = sm.OLS(y, X).fit()
                regression_summary = model.summary().as_html()

                # 3. สร้างกราฟ Correlation Heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0)
                plt.title('Correlation Analysis')
                
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()

                # 4. ระบบพยากรณ์ (ถ้ามีการกรอกตัวเลขเข้ามา)
                if 'val1' in request.form and request.form['val1']:
                    v1 = float(request.form.get('val1', 0))
                    v2 = float(request.form.get('val2', 0))
                    v3 = float(request.form.get('val3', 0))
                    # คำนวณตามสมการ: Y = const + b1X1 + b2X2 + b3X3
                    input_data = [1, v1, v2, v3]
                    prediction = model.predict(input_data)[0]

            except Exception as e:
                regression_summary = f"<div class='alert alert-danger'>Error: {str(e)}</div>"

    return render_template('index.html', 
                           plot_url=plot_url, 
                           correlation_table=correlation_html, 
                           regression_table=regression_summary,
                           prediction=prediction)

if __name__ == "__main__":
    # รองรับ Port สำหรับ Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
