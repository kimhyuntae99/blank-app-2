import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from fpdf import FPDF
import base64

# 샘플 건강 데이터 생성


import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import base64

st.set_page_config(page_title="BioData View", layout="centered")
st.title("🏃‍♂️ BioData View - 건강데이터 분석")
st.write("운동량(운동거리, 칼로리), 키, 체중, BMI를 입력하면 기술통계, 상관분석, 회귀분석, 시각화를 보여주는 앱입니다.")

# 데이터 입력
st.subheader("1. 데이터 입력")
st.write("학생별로 운동거리, 칼로리, 체중, BMI를 직접 입력하세요.")

num_students = st.number_input("학생 수", min_value=1, max_value=30, value=5)
student_data = []
for i in range(int(num_students)):
    st.markdown(f"**학생 {i+1}**")
    distance = st.number_input(f"운동거리(km) - 학생 {i+1}", min_value=0.0, max_value=100.0, value=0.0, key=f"distance_{i}")
    calorie = st.number_input(f"칼로리(kcal) - 학생 {i+1}", min_value=0.0, max_value=2000.0, value=0.0, key=f"calorie_{i}")
    height = st.number_input(f"키(cm) - 학생 {i+1}", min_value=0.0, max_value=250.0, value=0.0, key=f"height_{i}")
    weight = st.number_input(f"체중(kg) - 학생 {i+1}", min_value=0.0, max_value=200.0, value=0.0, key=f"weight_{i}")
    bmi = 0.0
    if height > 0:
        bmi = round(weight / ((height/100)**2), 2)
    st.write(f"BMI 자동 계산: {bmi}")
    student_data.append({
        '운동거리(km)': distance,
        '칼로리(kcal)': calorie,
        '키(cm)': height,
        '체중(kg)': weight,
        'BMI': bmi
    })
data = pd.DataFrame(student_data)

if len(data) > 0:
    st.subheader("2. 기술통계 요약")
    st.write(data.describe().T)

    st.subheader("3. 변수 선택 및 시각화")
    num_cols = [c for c in data.columns if data[c].dtype != 'O']
    x_var = st.selectbox("X축 변수", num_cols, index=0)
    y_var = st.selectbox("Y축 변수", num_cols, index=1 if len(num_cols)>1 else 0)
    chart_type = st.selectbox("그래프 종류", ["산점도", "히스토그램", "박스플롯"])

    # Plotly 시각화 (Streamlit에서만 사용)
    if chart_type == "산점도":
        fig_plotly = px.scatter(data, x=x_var, y=y_var, trendline="ols", title=f"{x_var} vs {y_var} 산점도 및 회귀선")
        st.plotly_chart(fig_plotly, use_container_width=True)
    elif chart_type == "히스토그램":
        fig_plotly = px.histogram(data, x=x_var, nbins=15, title=f"{x_var} 히스토그램")
        st.plotly_chart(fig_plotly, use_container_width=True)
    elif chart_type == "박스플롯":
        fig_plotly = px.box(data, y=x_var, title=f"{x_var} 박스플롯")
        st.plotly_chart(fig_plotly, use_container_width=True)

    # PDF용 matplotlib 그래프 생성 함수
    def get_matplotlib_image(data, x_var, y_var, chart_type):
        plt.figure(figsize=(5,4))
        img_buf = BytesIO()
        if chart_type == "산점도":
            plt.scatter(data[x_var], data[y_var], color='blue')
            # 회귀선
            if len(data[x_var]) > 1:
                m, b = np.polyfit(data[x_var], data[y_var], 1)
                plt.plot(data[x_var], m*data[x_var]+b, color='red')
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.title(f"{x_var} vs {y_var} 산점도 및 회귀선")
        elif chart_type == "히스토그램":
            plt.hist(data[x_var], bins=15, color='skyblue', edgecolor='black')
            plt.xlabel(x_var)
            plt.title(f"{x_var} 히스토그램")
        elif chart_type == "박스플롯":
            plt.boxplot(data[x_var].dropna())
            plt.ylabel(x_var)
            plt.title(f"{x_var} 박스플롯")
        plt.tight_layout()
        plt.savefig(img_buf, format='png')
        plt.close()
        img_buf.seek(0)
        return img_buf

    st.subheader("4. 상관분석 및 회귀분석 결과")
    if x_var != y_var:
        corr = data[x_var].corr(data[y_var])
        st.write(f"상관계수: {corr:.2f}")
        # 상관분석 자동 해석
        if abs(corr) > 0.7:
            level = '매우 강함'
        elif abs(corr) > 0.4:
            level = '상당히 강함'
        elif abs(corr) > 0.2:
            level = '약함'
        else:
            level = '거의 없음'
        direction = '양의' if corr > 0 else '음의'
        summary = f"{x_var}와 {y_var}의 상관계수는 {corr:.2f}로, {direction} 방향의 {level} 상관관계가 있습니다."
        if direction == '음의':
            summary += f" 예: {x_var}이 증가할수록 {y_var}가 감소하는 경향을 보임."
        else:
            summary += f" 예: {x_var}이 증가할수록 {y_var}도 증가하는 경향을 보임."
        st.write(f"해석: {summary}")

        # 회귀분석
        X = data[x_var]
        Y = data[y_var]
        X_const = sm.add_constant(X)
        model = sm.OLS(Y, X_const).fit()
        coef = model.params[x_var]
        intercept = model.params['const']
        r2 = model.rsquared
        pval = model.pvalues[x_var]
        st.write("회귀분석 결과 요약:")
        if pval < 0.05:
            sig = "통계적으로 유의함"
        else:
            sig = "통계적으로 유의하지 않음"
        if coef > 0:
            reg_dir = "양의"
        else:
            reg_dir = "음의"
        reg_summary = f"{x_var}가 1 증가할 때 {y_var}는 {coef:.2f}만큼 {reg_dir} 방향으로 변화합니다. (절편: {intercept:.2f}, 결정계수: {r2:.2f}, p값: {pval:.3f}, {sig})"
        st.write(reg_summary)
        st.write("상세 회귀분석 결과표:")
        st.write(model.summary())
    else:
        st.write("서로 다른 두 변수를 선택하세요.")

    st.subheader("5. BMI 25 달성을 위한 목표 계산기 및 운동 계획안 작성")
    st.write("상관분석과 회귀분석 결과를 참고하여, BMI 25를 달성하기 위한 자신의 운동 계획안을 작성해보세요.")

    # BMI 25 달성 계산기
    st.markdown("---")
    st.markdown("**BMI 25 달성을 위한 목표 계산기**")
    calc_height = st.number_input("계산용 키(cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    current_weight = st.number_input("현재 체중(kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1)
    target_bmi = 25.0
    target_weight = round(target_bmi * ((calc_height/100)**2), 1)
    weight_diff = round(current_weight - target_weight, 1)
    st.write(f"목표 체중: {target_weight} kg (BMI 25 기준)")
    if weight_diff > 0:
        st.write(f"감량 필요 체중: {weight_diff} kg")
        # 운동별 칼로리 소모량(1km 기준): 걷기 50kcal, 조깅 70kcal, 달리기 100kcal
        total_kcal = int(weight_diff * 7700)
        walk_km = round(total_kcal / 50, 1)
        jog_km = round(total_kcal / 70, 1)
        run_km = round(total_kcal / 100, 1)
        st.write(f"필요 소모 칼로리: {total_kcal} kcal")
        st.write(f"총 필요거리: 걷기 {walk_km} km, 조깅 {jog_km} km, 달리기 {run_km} km")

        st.markdown("**기간별 일일 운동 계획**")
        period_weeks = st.number_input("목표 달성 기간(주)", min_value=1, max_value=52, value=4)
        period_days = period_weeks * 7
        st.write(f"총 기간: {period_weeks}주 ({period_days}일)")
        st.write(f"일일 걷기: {round(walk_km/period_days,1)} km / 일일 조깅: {round(jog_km/period_days,1)} km / 일일 달리기: {round(run_km/period_days,1)} km")
    else:
        st.write("이미 BMI 25 이하입니다!")

    st.markdown("---")
    student_name = st.text_input("이름(필수)")
    report_text = st.text_area("BMI 25 달성을 위한 나의 운동 계획안", height=200)
    if st.button("계획안 제출 및 저장"):
        if student_name and report_text:
            import os
            import csv
            save_path = "student_reports.csv"
            file_exists = os.path.isfile(save_path)
            with open(save_path, "a", encoding="utf-8", newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["이름", "BMI 25 달성을 위한 운동 계획안"])
                writer.writerow([student_name, report_text])
            st.success("계획안이 저장되었습니다! 교사는 student_reports.csv 파일을 엑셀로 열어볼 수 있습니다.")
        else:
            st.warning("이름과 계획안을 모두 입력하세요.")

    # PDF 다운로드 기능
    if student_name and report_text:
        from fpdf import FPDF
        import base64
        pdf = FPDF()
        pdf.add_page()
        try:
            pdf.add_font('Nanum', '', 'NanumGothic.ttf', uni=True)
            font_name = 'Nanum'
        except:
            font_name = 'Arial'
        pdf.set_font(font_name, '', 16)
        pdf.cell(0, 10, 'BMI 25 달성을 위한 운동 계획안', ln=True, align='C')
        pdf.set_font(font_name, '', 12)
        pdf.cell(0, 10, f'이름: {student_name}', ln=True)
        pdf.multi_cell(0, 10, report_text)
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        b64 = base64.b64encode(pdf_output.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="BMI25_Plan_{student_name}.pdf">PDF 계획안 다운로드</a>'
        st.markdown(href, unsafe_allow_html=True)

        plot_imgs = []
