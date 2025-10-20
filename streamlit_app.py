import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from fpdf import FPDF
import base64
import os
import urllib.request

# 도움: 작업공간에 한글 TTF가 없으면 공용 raw GitHub에서 NanumGothic.ttf를 자동으로 내려받습니다.
def ensure_nanum_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return font_path
    # 후보 URL 목록: 실패할 수 있으므로 여러 URL을 시도
    candidate_urls = [
        'https://github.com/naver/nanumfont/raw/master/TTF/NanumGothic.ttf',
        'https://github.com/naver/nanumfont/raw/main/TTF/NanumGothic.ttf',
        'https://github.com/naver/nanumfont/raw/master/Fonts/NanumGothic.ttf',
        'https://raw.githubusercontent.com/kimhyuntae99/nanumfont/main/NanumGothic.ttf'
    ]
    for url in candidate_urls:
        try:
            urllib.request.urlretrieve(url, font_path)
            # quick check
            if os.path.exists(font_path) and os.path.getsize(font_path) > 1000:
                return font_path
        except Exception:
            continue
    return None

# 시도: 초기 로드 시 폰트가 없으면 받아보자 (실패해도 앱은 계속 동작)
try:
    ensure_nanum_font()
except Exception:
    pass

# PDF 생성 헬퍼 함수
def create_pdf_bytes(student_name: str, plan_text: str, summary: dict, mix_summary: dict, include_chart_bytes: bytes=None) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # 한글 폰트 시도: 작업공간에 NanumGothic.ttf가 있으면 등록
    have_unicode_font = False
    try:
        import os
        font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
        if os.path.exists(font_path):
            pdf.add_font('NanumGothic', '', font_path, uni=True)
            pdf.set_font('NanumGothic', size=12)
            have_unicode_font = True
        else:
            pdf.set_font('Arial', size=12)
    except Exception:
        pdf.set_font('Arial', size=12)

    # 안전한 텍스트 출력: 유니코드 폰트가 없으면 non-latin 문자를 대체
    def safe_text(s: str) -> str:
        if have_unicode_font:
            return str(s)
        try:
            return str(s).encode('latin-1', 'replace').decode('latin-1')
        except Exception:
            return ''.join(ch if ord(ch) < 256 else '?' for ch in str(s))

    pdf.cell(0, 10, safe_text('건강 리포트'), ln=1)
    if student_name:
        pdf.cell(0, 8, safe_text(f'이름: {student_name}'), ln=1)
    pdf.ln(2)

    # 요약 정보
    pdf.set_font('', size=11)
    pdf.multi_cell(0, 6, safe_text('=== BMI 목표 요약 ==='))
    for k, v in summary.items():
        pdf.multi_cell(0, 6, safe_text(f'{k}: {v}'))
    pdf.ln(2)

    # 혼합 운동 요약
    pdf.multi_cell(0, 6, safe_text('=== 혼합 운동 요약 ==='))
    for k, v in mix_summary.items():
        pdf.multi_cell(0, 6, safe_text(f'{k}: {v}'))
    pdf.ln(2)

    # 포함된 차트 이미지 (선택)
    if include_chart_bytes:
        try:
            # FPDF requires saving to a temp file or using image from BytesIO via pillow; use temp file
            import tempfile
            from PIL import Image
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            tmp.write(include_chart_bytes)
            tmp.flush()
            tmp.close()
            pdf.image(tmp.name, w=150)
        except Exception:
            pass

    pdf.ln(4)
    pdf.multi_cell(0, 6, safe_text('=== 작성한 건강 리포트 ==='))
    if plan_text:
        # 리포트 텍스트를 여러 줄로 넣기
        for line in plan_text.splitlines():
            pdf.multi_cell(0, 6, safe_text(line))
    else:
        pdf.multi_cell(0, 6, safe_text('작성된 리포트가 없습니다.'))

    # fpdf.output(dest='S') may attempt latin-1 encoding which fails for non-latin chars.
    # Safer approach: write to a temp file in binary and return its bytes.
    try:
        import tempfile
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        tmp_path = tmpf.name
        tmpf.close()
        pdf.output(tmp_path)
        with open(tmp_path, 'rb') as f:
            pdf_bytes = f.read()
        # try to cleanup the temp file
        try:
            import os
            os.remove(tmp_path)
        except Exception:
            pass
        return pdf_bytes
    except Exception as e:
        # If writing to temp file fails, raise a clear error instead of attempting
        # to encode the PDF string with latin-1 (which fails for non-latin text).
        raise RuntimeError(f"PDF 생성 실패: {e}") from e

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
st.write("운동량(운동거리, 칼로리), 키, 체중, BMI를 입력하거나 템플릿으로 업로드하여 기술통계, 상관분석, 회귀분석, 시각화를 할 수 있는 앱입니다.")

# 탭 구성: 수동 입력 / 템플릿 업로드 / 분석
tab_manual, tab_upload, tab_analysis = st.tabs(["수동 입력", "템플릿 업로드", "분석"])

if 'current_data' not in st.session_state:
    st.session_state['current_data'] = None

with tab_manual:
    st.header("수동 입력")
    st.write("학생별로 운동거리, 칼로리, 키, 체중을 입력하면 BMI는 자동 계산됩니다.")
    num_students = st.number_input("학생 수", min_value=1, max_value=30, value=5, key='num_students')
    student_data = []
    for i in range(int(num_students)):
        st.markdown(f"**학생 {i+1}**")
        distance = st.number_input(f"운동거리(km) - 학생 {i+1}", min_value=0.0, max_value=100.0, value=0.0, key=f"distance_{i}")
        calorie = st.number_input(f"칼로리(kcal) - 학생 {i+1}", min_value=0.0, max_value=2000.0, value=0.0, key=f"calorie_{i}")
        height = st.number_input(f"키(cm) - 학생 {i+1}", min_value=0.0, max_value=250.0, value=170.0, key=f"height_{i}")
        weight = st.number_input(f"체중(kg) - 학생 {i+1}", min_value=0.0, max_value=200.0, value=65.0, key=f"weight_{i}")
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
    manual_df = pd.DataFrame(student_data)
    st.session_state['current_data'] = manual_df
    st.success("수동 입력 데이터가 준비되었습니다. '분석' 탭으로 이동하세요.")

with tab_upload:
    st.header("템플릿 업로드")
    st.write("아래 템플릿을 내려받아 작성한 뒤 업로드하세요. (CSV 또는 Excel)")
    template_df = pd.DataFrame({
        '운동거리(km)': [0], '칼로리(kcal)': [0], '키(cm)': [170], '체중(kg)': [65], 'BMI': [0]
    })
    csv_bytes = template_df.to_csv(index=False).encode('utf-8')
    st.download_button("템플릿 CSV 다운로드", data=csv_bytes, file_name="biodata_template.csv", mime="text/csv")
    try:
        xlsx_buf = BytesIO()
        template_df.to_excel(xlsx_buf, index=False)
        xlsx_buf.seek(0)
        st.download_button("템플릿 XLSX 다운로드", data=xlsx_buf, file_name="biodata_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.info("XLSX 템플릿 다운로드는 환경에 따라 제한될 수 있습니다. CSV를 사용하세요.")

    uploaded = st.file_uploader("완성된 템플릿 업로드 (CSV 또는 XLSX)", type=["csv", "xlsx"])
    if uploaded is not None:
        # read raw bytes once, then try decoding with several encodings for CSV
        try:
            raw = uploaded.read()
            uploaded_df = None
            if uploaded.name.lower().endswith('.csv'):
                encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'latin1']
                last_err = None
                for enc in encodings_to_try:
                    try:
                        uploaded_df = pd.read_csv(BytesIO(raw), encoding=enc)
                        st.success(f"CSV 파일을 '{enc}' 인코딩으로 읽었습니다.")
                        break
                    except Exception as e_csv:
                        last_err = e_csv
                        # try next encoding
                if uploaded_df is None:
                    # raise the last error to be handled below
                    raise last_err
            else:
                # try reading excel with openpyxl first, fallback to default engine
                try:
                    uploaded_df = pd.read_excel(BytesIO(raw), engine='openpyxl')
                except Exception:
                    uploaded_df = pd.read_excel(BytesIO(raw))

            st.write("업로드된 데이터 미리보기")
            st.dataframe(uploaded_df.head())
            # 결측치율은 자동으로 화면에 표시하지 않음
            if st.button("분석 데이터로 사용"):
                # 자동으로 BMI 계산
                if 'BMI' not in uploaded_df.columns or uploaded_df['BMI'].isnull().all():
                    if '키(cm)' in uploaded_df.columns and '체중(kg)' in uploaded_df.columns:
                        uploaded_df['BMI'] = (uploaded_df['체중(kg)'] / ((uploaded_df['키(cm)']/100)**2)).round(2)
                st.session_state['current_data'] = uploaded_df
                st.success("업로드 데이터가 분석 데이터로 설정되었습니다. '분석' 탭으로 이동하세요.")
        except UnicodeDecodeError:
            st.error("파일을 읽는 중 인코딩 오류가 발생했습니다. CSV 파일 인코딩을 UTF-8, CP949 또는 EUC-KR로 변환한 뒤 다시 업로드해 주세요.")
        except Exception as e:
            # Provide a friendly hint for common encoding issues
            msg = str(e)
            if 'utf-8' in msg or 'codec' in msg:
                st.error("파일을 읽는 중 인코딩 오류가 발생했습니다. CSV 파일을 UTF-8 또는 CP949(윈도우용)로 저장한 뒤 다시 업로드해 주세요.")
            else:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}. CSV로 변환하여 다시 업로드해 주세요.")

with tab_analysis:
    st.header("분석")
    data = st.session_state.get('current_data')
    if data is None or data.empty:
        st.info("분석할 데이터가 없습니다. '수동 입력' 또는 '템플릿 업로드' 탭에서 데이터를 준비하세요.")
    else:
        st.subheader("데이터 미리보기")
        st.dataframe(data)
    # 결측치율 요약은 화면에 자동 표시하지 않음

        # 기존 분석 코드 사용 (변수 선택, 시각화, 상관/회귀, BMI 계산기 등.)
        num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        st.markdown("**시각화 옵션**")
        show_all_bars = st.checkbox("모든 수치형 변수 막대그래프로 보기")
        if show_all_bars:
            # x labels: 이름 컬럼이 있으면 사용, 아니면 인덱스
            if '이름' in data.columns:
                labels = data['이름'].astype(str)
            elif 'name' in data.columns:
                labels = data['name'].astype(str)
            else:
                labels = data.index.astype(str)
            st.info(f"{len(num_cols)}개의 수치형 변수를 막대그래프로 표시합니다.")
            for col in num_cols:
                fig_bar = px.bar(data, x=labels, y=col, title=f"{col} - 값 분포 (행별)")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            x_var = st.selectbox("X축 변수", num_cols, index=0)
            y_var = st.selectbox("Y축 변수", num_cols, index=1 if len(num_cols)>1 else 0)
            chart_type = st.selectbox("그래프 종류", ["산점도", "히스토그램", "박스플롯"], key='chart_main')

            if chart_type == "산점도":
                fig_plotly = px.scatter(data, x=x_var, y=y_var, trendline="ols", title=f"{x_var} vs {y_var} 산점도 및 회귀선")
                st.plotly_chart(fig_plotly, use_container_width=True)
            elif chart_type == "히스토그램":
                fig_plotly = px.histogram(data, x=x_var, nbins=15, title=f"{x_var} 히스토그램")
                st.plotly_chart(fig_plotly, use_container_width=True)
            elif chart_type == "박스플롯":
                fig_plotly = px.box(data, y=x_var, title=f"{x_var} 박스플롯")
                st.plotly_chart(fig_plotly, use_container_width=True)

        # reuse analysis (correlation and regression)
        st.subheader("상관분석 및 회귀분석 결과")
        if x_var != y_var:
            corr = data[x_var].corr(data[y_var])
            st.write(f"상관계수: {corr:.2f}")
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
            st.write(f"해석: {summary}")

            X = data[x_var]
            Y = data[y_var]
            X_const = sm.add_constant(X)
            model = sm.OLS(Y, X_const).fit()
            coef = model.params[x_var]
            intercept = model.params['const']
            r2 = model.rsquared
            pval = model.pvalues[x_var]
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

        # BMI 25 달성 계산기 (분석 탭 하단)
        st.markdown("---")
        st.subheader("BMI 25 달성을 위한 목표 계산기 및 운동 계획")
        st.write("데이터에서 평균값을 자동으로 불러오거나, 직접 값을 입력하여 계산할 수 있습니다.")
        use_data_avg = False
        try:
            avg_height = round(data['키(cm)'].mean(),1) if '키(cm)' in data.columns else 170.0
            avg_weight = round(data['체중(kg)'].mean(),1) if '체중(kg)' in data.columns else 65.0
        except Exception:
            avg_height = 170.0
            avg_weight = 65.0
        if st.checkbox("데이터 평균값 사용 (있을 경우)"):
            calc_height = st.number_input("계산용 키(cm)", min_value=100.0, max_value=250.0, value=float(avg_height), step=0.1)
            current_weight = st.number_input("현재 체중(kg)", min_value=30.0, max_value=200.0, value=float(avg_weight), step=0.1)
        else:
            calc_height = st.number_input("계산용 키(cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
            current_weight = st.number_input("현재 체중(kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1)

        target_bmi = 25.0
        target_weight = round(target_bmi * ((calc_height/100)**2), 1)
        weight_diff = round(current_weight - target_weight, 1)
        st.write(f"목표 체중(BMI {target_bmi} 기준): {target_weight} kg")
        if weight_diff > 0:
            st.write(f"감량 필요 체중: {weight_diff} kg")
            # 운동별 칼로리 소모량(1km 기준): 걷기 50kcal, 조깅 70kcal, 달리기 100kcal
            total_kcal = int(weight_diff * 7700)
            walk_km = round(total_kcal / 50, 1)
            jog_km = round(total_kcal / 70, 1)
            run_km = round(total_kcal / 100, 1)
            st.write(f"필요 소모 칼로리(총): {total_kcal} kcal")
            st.write(f"총 필요거리: 걷기 {walk_km} km, 조깅 {jog_km} km, 달리기 {run_km} km")

            st.markdown("**기간별 일일 운동 계획**")
            period_weeks = st.number_input("목표 달성 기간(주)", min_value=1, max_value=52, value=4, key='period_weeks')
            period_days = period_weeks * 7
            st.write(f"총 기간: {period_weeks}주 ({period_days}일)")
            st.write(f"일일 걷기: {round(walk_km/period_days,1)} km / 일일 조깅: {round(jog_km/period_days,1)} km / 일일 달리기: {round(run_km/period_days,1)} km")

            st.markdown("**혼합 운동 추천 예시**")
            st.write("다양한 비율로 혼합하여 진행할 수 있습니다. 아래는 예시 비율입니다.")
            # 기본 비율: 걷기50%, 조깅30%, 달리기20%
            mix_walk_pct = st.slider("걷기 비율(%)", 0, 100, 50)
            mix_jog_pct = st.slider("조깅 비율(%)", 0, 100, 30)
            mix_run_pct = st.slider("달리기 비율(%)", 0, 100, 20)
            total_pct = mix_walk_pct + mix_jog_pct + mix_run_pct
            if total_pct == 0:
                st.warning("비율의 합이 0입니다. 비율을 조정하세요.")
            else:
                walk_mix = round(walk_km * (mix_walk_pct/100),1)
                jog_mix = round(jog_km * (mix_jog_pct/100),1)
                run_mix = round(run_km * (mix_run_pct/100),1)
                st.write(f"혼합 추천(비율 반영): 걷기 {walk_mix} km + 조깅 {jog_mix} km + 달리기 {run_mix} km (총 {round(walk_mix + jog_mix + run_mix,1)} km)")
                st.write(f"일일 혼합 운동(기간 {period_days}일 기준): 걷기 {round(walk_mix/period_days,1)} km / 조깅 {round(jog_mix/period_days,1)} km / 달리기 {round(run_mix/period_days,1)} km")
        else:
            st.info("현재 체중이 이미 BMI 25 이하이거나 감량이 필요하지 않습니다.")
        st.markdown("---")
        st.subheader("건강 리포트 작성 (BMI 25 달성을 위한 운동 계획안)")
        st.markdown("**작성 참고 예시**")
        st.info("예시: 나는 키가 170cm, 몸무게 80kg, BMI 값이 26이다. 그래서 BMI 25를 달성하기 위해 16주동안 걷기 2.3 km, 조깅 1.6 km, 달리기 1.0 km 할 것이다.")
        student_name = st.text_input("이름(선택)")
        plan_text = st.text_area("BMI 25 달성을 위한 나의 운동 계획안", height=200)
        if plan_text:
            st.success("계획안이 입력되었습니다. 필요하면 내용을 복사하거나 PDF로 내보내도록 요청하세요.")

        # PDF 다운로드 준비
        try:
            summary = {
                '현재 키(cm)': f"{calc_height}",
                '현재 체중(kg)': f"{current_weight}",
                '목표 체중(kg)': f"{target_weight}",
                '감량 필요(kg)': f"{weight_diff}",
                '필요 총 소모 칼로리(kcal)': f"{total_kcal if 'total_kcal' in locals() else 0}",
            }
        except Exception:
            summary = {}

        try:
            mix_summary = {
                '혼합 걷기 총(km)': f"{walk_mix if 'walk_mix' in locals() else 0}",
                '혼합 조깅 총(km)': f"{jog_mix if 'jog_mix' in locals() else 0}",
                '혼합 달리기 총(km)': f"{run_mix if 'run_mix' in locals() else 0}",
                '일일 걷기(km)': f"{round((walk_mix/period_days),1) if 'walk_mix' in locals() and period_days>0 else 0}",
                '일일 조깅(km)': f"{round((jog_mix/period_days),1) if 'jog_mix' in locals() and period_days>0 else 0}",
                '일일 달리기(km)': f"{round((run_mix/period_days),1) if 'run_mix' in locals() and period_days>0 else 0}",
            }
        except Exception:
            mix_summary = {}

        # (선택) 현재 플롯 이미지를 PDF에 포함하려면 matplotlib로 그려서 바이트 생성 가능
        chart_bytes = None

        pdf_bytes = create_pdf_bytes(student_name=student_name or '', plan_text=plan_text or '', summary=summary, mix_summary=mix_summary, include_chart_bytes=chart_bytes)
        st.download_button(label="건강 리포트 PDF 다운로드", data=pdf_bytes, file_name=f"biodata_report_{student_name or 'student'}.pdf", mime='application/pdf')
