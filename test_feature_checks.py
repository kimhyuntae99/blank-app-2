import pandas as pd
import numpy as np
from io import BytesIO
import tempfile
import os

print('Preparing sample data...')
# Create sample DataFrame with 12 rows to avoid small-sample warnings
np.random.seed(42)
N = 12
height = np.random.normal(170, 7, N).round(1)
weight = np.random.normal(68, 8, N).round(1)
distance = np.random.exponential(1.5, N).round(2)  # km
calorie = (distance * 70 + np.random.normal(0, 30, N)).round(1)

bmi = (weight / ((height/100)**2)).round(2)

sample_df = pd.DataFrame({
    '운동거리(km)': distance,
    '칼로리(kcal)': calorie,
    '키(cm)': height,
    '체중(kg)': weight,
    'BMI': bmi
})

print('\nSample data:')
print(sample_df)

# Test BMI auto-calculation
print('\nVerifying BMI calculation from 키/체중...')
calculated_bmi = (sample_df['체중(kg)'] / ((sample_df['키(cm)']/100)**2)).round(2)
if calculated_bmi.equals(sample_df['BMI']):
    print('BMI 계산 일치')
else:
    print('BMI 불일치 - 차이:')
    print((calculated_bmi - sample_df['BMI']).abs().sum())

# Test correlation and regression for a few pairs
from scipy import stats
import statsmodels.api as sm

pairs = [
    ('운동거리(km)', '체중(kg)'),
    ('칼로리(kcal)', 'BMI'),
    ('키(cm)', 'BMI')
]

for xcol, ycol in pairs:
    print('\n---')
    print(f'Pair: {xcol} -> {ycol}')
    pair = sample_df[[xcol, ycol]].dropna()
    print(f'observations: {len(pair)}')
    if len(pair) < 2:
        print('데이터가 부족하여 계산 불가')
        continue
    x = pair[xcol]
    y = pair[ycol]
    try:
        r, p = stats.pearsonr(x, y)
        print(f'Pearson r = {r:.4f}, p = {p:.4f}')
    except Exception as e:
        print('Pearson 계산 실패:', e)
        r = x.corr(y)
        p = np.nan
        print(f'Fallback corr = {r:.4f}')

    # OLS
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = model.params.get(xcol, np.nan)
    intercept = model.params.get('const', np.nan)
    r2 = model.rsquared
    pval = model.pvalues.get(xcol, np.nan)
    stderr = model.bse.get(xcol, np.nan)
    print('OLS results:')
    print(f' slope = {slope:.4f}, intercept = {intercept:.4f}')
    print(f' R^2 = {r2:.4f}, slope p = {pval:.4f}, slope stderr = {stderr:.4f}')

# Test template encoding: write utf-8-sig and cp949 and read back
print('\nTesting template CSV write/read (utf-8-sig & cp949)')
with tempfile.TemporaryDirectory() as td:
    utf_path = os.path.join(td, 'template_utf8.csv')
    cp949_path = os.path.join(td, 'template_cp949.csv')
    sample_df.to_csv(utf_path, index=False, encoding='utf-8-sig')
    sample_df.to_csv(cp949_path, index=False, encoding='cp949', errors='replace')
    print('Wrote files:', utf_path, cp949_path)
    # Read back
    df_utf = pd.read_csv(utf_path)
    df_cp949 = pd.read_csv(cp949_path, encoding='cp949')
    print('Read back shapes:', df_utf.shape, df_cp949.shape)
    # quick compare head
    print('\nUTF sample:')
    print(df_utf.head().to_string(index=False))
    print('\nCP949 sample:')
    print(df_cp949.head().to_string(index=False))

print('\nAll tests finished.')
