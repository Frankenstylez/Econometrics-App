import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle
import os
from sklearn.linear_model import LinearRegression

# ========================================================
# 1. ส่วนตั้งค่า (CONFIGURATION) - แก้ไขเฉพาะตรงนี้เมื่อเปลี่ยนไฟล์
# ========================================================
FILE_NAME = 'Data for econometrics.xlsx'
X_COLS = ['X1 : Ad Expense Thousand THB', 'X2 : GDP (Billion THB)', 'X3 : CPI']
Y_COL = 'Y : Revennue Thousand THB'

# ========================================================
# 2. การโหลดข้อมูลและเตรียมข้อมูล
# ========================================================
try:
    if not os.path.exists(FILE_NAME):
        raise FileNotFoundError(f"หาไฟล์ '{FILE_NAME}' ไม่เจอในโฟลเดอร์โปรเจกต์")

    # อ่านไฟล์ Excel
    df = pd.read_excel(FILE_NAME)
    df.columns = df.columns.str.strip()  # ตัดช่องว่างชื่อคอลัมน์

    # ลบแถวที่มีค่าว่างในตัวแปรที่ใช้งาน
    df_clean = df[X_COLS + [Y_COL]].dropna()
    print("✅ 1. โหลดและเตรียมข้อมูลเรียบร้อย")

    # ========================================================
    # 3. วิเคราะห์ REGRESSION (OLS) และเซฟผลลัพธ์
    # ========================================================
    X = df_clean[X_COLS]
    Y = df_clean[Y_COL]
    X_with_const = sm.add_constant(X)  # เพิ่มค่า Constant (Intercept)

    model = sm.OLS(Y, X_with_const).fit()

    # แสดงผลหน้าจอ
    print("\n--- สรุปผลทางสถิติ (Regression Summary) ---")
    print(model.summary())

    # เซฟตารางสรุปเป็นไฟล์ Text
    with open('regression_summary.txt', 'w', encoding='utf-8') as f:
        f.write(model.summary().as_text())

    # เซฟตัวโมเดลไว้ใช้พยากรณ์ในอนาคต
    with open('econometrics_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("\n✅ 2. วิเคราะห์และเซฟไฟล์สถิติ (.txt, .pkl) เรียบร้อย")

    # ========================================================
    # 4. การสร้างกราฟแยก 3 ตัวแปร (Ad Expense, GDP, CPI)
    # ========================================================
    # สร้างพื้นที่กราฟ 1 แถว 3 คอลัมน์
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # สีน้ำเงิน, เขียว, แดง

    for i, col_name in enumerate(X_COLS):
        # เตรียมข้อมูลรายตัวแปร
        X_sub = df_clean[[col_name]].values
        Y_sub = df_clean[Y_COL].values

        # คำนวณเส้นตรง (Linear Regression)
        line_model = LinearRegression().fit(X_sub, Y_sub)
        Y_pred = line_model.predict(X_sub)

        # ค่าสมการ
        slope = line_model.coef_[0]
        intercept = line_model.intercept_
        r2 = line_model.score(X_sub, Y_sub)

        # วาดกราฟ
        axes[i].scatter(X_sub, Y_sub, color=colors[i], alpha=0.5, label='Actual Data')
        axes[i].plot(X_sub, Y_pred, color='black', linestyle='--', linewidth=2, label='Trend Line')

        # ใส่ชื่อกราฟและป้ายกำกับ
        axes[i].set_title(f'Impact of {col_name.split(":")[0]}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel(col_name, fontsize=10)
        axes[i].set_ylabel(Y_COL, fontsize=10)

        # เขียนสมการลงบนกราฟ
        eq_label = f'$Y = {slope:.2f}X + ({intercept:.2f})$\n$R^2 = {r2:.3f}$'
        axes[i].annotate(eq_label, xy=(0.05, 0.85), xycoords='axes fraction',
                         fontsize=12, color='darkred', fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    # เซฟรูปกราฟ
    plt.savefig('econometrics_3_plots.png', dpi=300)
    print("✅ 3. สร้างและเซฟรูปกราฟ 'econometrics_3_plots.png' เรียบร้อย")

    # แสดงกราฟออกมา
    plt.show()

except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")