from fpdf import FPDF
import os

def create_pdf_bytes_min(student_name: str, plan_text: str, summary: dict, mix_summary: dict, include_chart_bytes: bytes=None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    have_unicode_font = False
    try:
        font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
        if os.path.exists(font_path):
            pdf.add_font('NanumGothic', '', font_path, uni=True)
            pdf.set_font('NanumGothic', size=12)
            have_unicode_font = True
        else:
            pdf.set_font('Arial', size=12)
    except Exception:
        pdf.set_font('Arial', size=12)

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
    pdf.set_font('', size=11)
    pdf.multi_cell(0, 6, safe_text('=== BMI 목표 요약 ==='))
    for k, v in summary.items():
        pdf.multi_cell(0, 6, safe_text(f'{k}: {v}'))
    pdf.ln(2)
    pdf.multi_cell(0, 6, safe_text('=== 혼합 운동 요약 ==='))
    for k, v in mix_summary.items():
        pdf.multi_cell(0, 6, safe_text(f'{k}: {v}'))
    pdf.ln(2)
    pdf.ln(4)
    pdf.multi_cell(0, 6, safe_text('=== 작성한 건강 리포트 ==='))
    if plan_text:
        for line in plan_text.splitlines():
            pdf.multi_cell(0, 6, safe_text(line))
    else:
        pdf.multi_cell(0, 6, safe_text('작성된 리포트가 없습니다.'))
    import tempfile
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    tmp_path = tmpf.name
    tmpf.close()
    pdf.output(tmp_path)
    with open(tmp_path, 'rb') as f:
        pdf_bytes = f.read()
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return pdf_bytes

if __name__ == '__main__':
    student_name = '홍길동'
    plan_text = '매일 걷기 30분\n조깅 20분\n달리기 10분'
    summary = {'현재 키(cm)': '170', '현재 체중(kg)': '80', '목표 체중(kg)': '72.25', '감량 필요(kg)': '7.75', '필요 총 소모 칼로리(kcal)': '59675'}
    mix_summary = {'혼합 걷기 총(km)': '30.0', '혼합 조깅 총(km)': '20.0', '혼합 달리기 총(km)': '10.0', '일일 걷기(km)': '0.5'}
    pdf_bytes = create_pdf_bytes_min(student_name=student_name, plan_text=plan_text, summary=summary, mix_summary=mix_summary)
    out = 'test_report_runner.pdf'
    with open(out, 'wb') as f:
        f.write(pdf_bytes)
    print('Wrote', out)
