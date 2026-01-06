from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import io, base64, os

app = Flask(__name__)
plt.switch_backend('Agg')

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    correlation_table = None
    regression_table = None
    prediction = None
    y_name = ""
    x_names = []

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            try:
                df = pd.read_excel(file)
                df.columns = df.columns.str.strip()
                all_cols = df.columns.tolist()

                y_list = [c for c in all_cols if 'Y' in c.upper()]
                y_name = y_list[0] if y_list else all_cols[-1]
                x_names = [c for c in all_cols if 'X' in c.upper() and c != y_name]
                if not x_names: x_names = [c for c in all_cols if c != y_name]

                df_clean = df[x_names + [y_name]].dropna()

                # --- คำนวณ Regression ---
                X = sm.add_constant(df_clean[x_names])
                y = df_clean[y_name]
                model = sm.OLS(y, X).fit()
                regression_table = model.summary().as_html()

                # --- สร้างกราฟ Scatter + Trend Line + Equation ---
                num_x = len(x_names)
                fig, axes = plt.subplots(1, num_x, figsize=(6*num_x, 5), squeeze=False)
                
                for i, x_col in enumerate(x_names):
                    # วาดจุดข้อมูล
                    sns.regplot(x=x_col, y=y_name, data=df_clean, ax=axes[0, i], 
                                line_kws={"color": "red", "lw": 2})
                    
                    # คำนวณสมการรายตัวแปร (Simple Regression สำหรับกราฟ)
                    x_sub = sm.add_constant(df_clean[x_col])
                    sub_model = sm.OLS(y, x_sub).fit()
                    intercept = sub_model.params[0]
                    slope = sub_model.params[1]
                    r2 = sub_model.rsquared
                    
                    # ใส่ข้อความสมการบนกราฟ
                    eq_text = f'Y = {slope:.2f}X + {intercept:.2f}\n$R^2$ = {r2:.3f}'
                    axes[0, i].set_title(f'Relation: {x_col}\n{eq_text}', fontsize=12, color='darkblue')

                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()

                # ระบบพยากรณ์
                inputs = [float(request.form.get(f'x_input_{i}', 0)) for i in range(len(x_names)) if request.form.get(f'x_input_{i}')]
                if len(inputs) == len(x_names):
                    prediction = model.predict([1] + inputs)[0]

            except Exception as e:
                regression_table = f"<div class='alert alert-danger'>{str(e)}</div>"

    return render_template('index.html', plot_url=plot_url, regression_table=regression_table, 
                           prediction=prediction, y_name=y_name, x_names=x_names)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
