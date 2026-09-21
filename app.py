import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title='Loan Portfolio Intelligence 2.0', page_icon='◈', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:#080d18;color:#e8eef8} section[data-testid="stSidebar"]{background:#0d1424}
.hero{padding:22px 26px;border:1px solid #263752;border-radius:18px;background:linear-gradient(135deg,#101b30,#0b1220);margin-bottom:18px}.hero-title{font-size:34px;font-weight:800;color:#f4f8ff}.hero-sub{color:#91a4c2;margin-top:6px}.kpi{background:#111e33;border:1px solid #263752;border-radius:16px;padding:16px;min-height:105px}.kpi-label{color:#93a7c5;font-size:12px;text-transform:uppercase;letter-spacing:1px}.kpi-value{color:#f4f8ff;font-size:27px;font-weight:800;margin-top:10px}.kpi-note{color:#6ee7d0;font-size:11px;margin-top:5px}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    for p in [Path('final_loan_portfolio.csv'),Path('data/final_loan_portfolio.csv'),Path('/content/final_loan_portfolio.csv'),Path('/mnt/data/final_loan_portfolio.csv')]:
        if p.exists():
            df=pd.read_csv(p); break
    else: return None
    numeric=['original_upb','original_interest_rate','credit_score','original_dti','original_ltv','original_cltv','ever_delinquent_flag','ever_seriously_delinquent_flag']
    for c in numeric:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df

def money(x): return '—' if pd.isna(x) else (f'${x/1e9:.2f}B' if abs(x)>=1e9 else f'${x/1e6:.1f}M')
df=load_data()
if df is None: st.error('final_loan_portfolio.csv not found. Put it in the repository root.'); st.stop()

st.markdown('<div class="hero"><div class="hero-title">Loan Portfolio Intelligence 2.0</div><div class="hero-sub">Executive analytics • Explainable risk screening • ML delinquency classification • Scenario analysis</div></div>',unsafe_allow_html=True)

st.sidebar.header('Portfolio Controls')
st.sidebar.caption('All views respond to these filters.')
states=sorted(df.property_state.dropna().astype(str).unique())
risks=sorted(df.combined_risk_indicator.dropna().astype(str).unique())
selected_states=st.sidebar.multiselect('Property state',states)
selected_risks=st.sidebar.multiselect('Analytical risk band',risks)
dti_max=st.sidebar.slider('Maximum DTI',0,65,65)
ltv_max=st.sidebar.slider('Maximum LTV',0,100,100)
filtered=df.copy()
if selected_states: filtered=filtered[filtered.property_state.astype(str).isin(selected_states)]
if selected_risks: filtered=filtered[filtered.combined_risk_indicator.astype(str).isin(selected_risks)]
filtered=filtered[filtered.original_dti.fillna(0)<=dti_max]
filtered=filtered[filtered.original_ltv.fillna(0)<=ltv_max]

loans=len(filtered); portfolio=filtered.original_upb.sum(); delinquency=filtered.ever_delinquent_flag.mean()*100 if loans else 0; serious=filtered.ever_seriously_delinquent_flag.mean()*100 if loans else 0
cols=st.columns(5)
for c,label,val,note in zip(cols,['Total loans','Portfolio value','Ever delinquent','Serious delinquency','Average credit score'],[f'{loans:,}',money(portfolio),f'{delinquency:.2f}%',f'{serious:.2f}%',f'{filtered.credit_score.mean():.1f}' if loans else '—'],['Filtered records','Original UPB • USD','Historical proxy','Historical proxy','Available scores']):
    c.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)

home, risk_tab, ml_tab, scenario_tab, data_tab = st.tabs(['Executive Dashboard','Risk Intelligence','ML Classification','Scenario Analysis','Data Explorer'])
with home:
    st.subheader('Business insights')
    if loans:
        top_state=filtered.groupby('property_state').size().sort_values(ascending=False).index[0]
        high_risk=filtered[filtered.combined_risk_indicator.astype(str).str.contains('High',na=False)]
        st.info(f'**Portfolio concentration:** {top_state} has the largest filtered loan count. **Risk exposure:** {len(high_risk):,} filtered loans fall in the project-defined high-risk indicator band. **Interpretation:** these are descriptive portfolio signals, not underwriting decisions.')
    c1,c2=st.columns(2)
    state=filtered.groupby('property_state',as_index=False).agg(loans=('loan_sequence_number','count'),value=('original_upb','sum')).sort_values('loans',ascending=False).head(15)
    c1.plotly_chart(px.bar(state,x='loans',y='property_state',orientation='h',title='Loan volume by state',template='plotly_dark'),use_container_width=True)
    purpose=filtered.groupby('loan_purpose',as_index=False).agg(loans=('loan_sequence_number','count'),delinquency=('ever_delinquent_flag','mean')); purpose['delinquency']*=100
    c2.plotly_chart(px.bar(purpose,x='loan_purpose',y='loans',color='delinquency',title='Purpose volume and delinquency intensity',template='plotly_dark'),use_container_width=True)
    st.subheader('Business actions to investigate')
    st.markdown('- Review concentrated state-level exposure before expanding similar portfolios.\n- Segment monitoring by DTI, LTV and credit-score bands.\n- Use the ML model as a research screening aid, not as a validated lending decision engine.')

with risk_tab:
    st.subheader('Risk matrix: credit score vs DTI')
    matrix=filtered.copy(); matrix['credit_band']=pd.cut(matrix.credit_score,[0,669,739,799,900],labels=['Below 670','670–739','740–799','800+'])
    matrix['dti_band']=pd.cut(matrix.original_dti,[-1,20,35,45,100],labels=['Low ≤20','Moderate 21–35','High 36–45','Very high >45'])
    heat=matrix.pivot_table(index='dti_band',columns='credit_band',values='ever_delinquent_flag',aggfunc='mean')*100
    st.dataframe(heat.round(2),use_container_width=True)
    st.caption('Cells show historical ever-delinquency percentage in this sample. Empty cells mean insufficient observations.')
    risk=filtered.groupby('combined_risk_indicator',as_index=False).agg(loans=('loan_sequence_number','count'),delinquency=('ever_delinquent_flag','mean')); risk['delinquency']*=100
    st.plotly_chart(px.bar(risk,x='combined_risk_indicator',y='delinquency',color='combined_risk_indicator',title='Observed delinquency by analytical risk band',template='plotly_dark'),use_container_width=True)

with ml_tab:
    st.subheader('Explainable ML: ever-delinquency classification')
    st.caption('Train/test split with origination attributes only. Performance is research-oriented and not production underwriting validation.')
    features=['credit_score','original_dti','original_ltv','original_cltv','original_interest_rate','original_upb','original_loan_term','number_of_borrowers','property_state','loan_purpose','occupancy_status','channel']
    features=[c for c in features if c in df.columns]; target='ever_delinquent_flag'
    model_df=df[features+[target]].copy().dropna(subset=[target]); X=model_df[features]; y=model_df[target].astype(int)
    if y.nunique()<2: st.warning('Target has only one class; ML metrics cannot be calculated.')
    else:
        cat=[c for c in features if X[c].dtype=='object']; num=[c for c in features if c not in cat]
        prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
        pipe=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,class_weight='balanced'))])
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
        mets=[accuracy_score(yte,pred),precision_score(yte,pred,zero_division=0),recall_score(yte,pred,zero_division=0),f1_score(yte,pred,zero_division=0),roc_auc_score(yte,prob)]
        cc=st.columns(5)
        for c,l,v in zip(cc,['Accuracy','Precision','Recall','F1','ROC-AUC'],mets): c.metric(l,f'{v:.3f}')
        st.write('Confusion matrix'); st.dataframe(pd.DataFrame(confusion_matrix(yte,pred),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1']))
        st.success('Model completed using the real project dataset. Metrics may change with class balance, random seed and feature selection.')

with scenario_tab:
    st.subheader('Assumption-based loss scenario')
    st.caption('This is not observed lender loss. It applies user-selected assumptions to the filtered portfolio.')
    stress=st.slider('Stress uplift to delinquency rate (%)',0.0,20.0,5.0,.5); recovery=st.slider('Assumed recovery rate (%)',0,100,60); exposure=filtered.original_upb.sum(); base=filtered.ever_delinquent_flag.mean() if loans else 0; stressed=min(base+stress/100,1); estimated_loss=exposure*stressed*(1-recovery/100)
    a,b,c=st.columns(3); a.metric('Base observed rate',f'{base*100:.2f}%'); b.metric('Stressed rate',f'{stressed*100:.2f}%'); c.metric('Illustrative loss',money(estimated_loss))
    st.warning('The stress uplift and recovery rate are assumptions. They are not supplied by Freddie Mac and should not be presented as actual realized loss.')

with data_tab:
    st.subheader('Filtered records')
    show=[c for c in ['loan_sequence_number','property_state','loan_purpose','original_upb','credit_score','original_dti','original_ltv','combined_risk_indicator','ever_delinquent_flag'] if c in filtered.columns]
    st.dataframe(filtered[show].head(1000),use_container_width=True,height=360)
    st.download_button('Download filtered CSV',filtered.to_csv(index=False),'filtered_loan_portfolio.csv','text/csv')

st.caption('Source: Freddie Mac 2020 sample. Delinquency flags and risk bands are analytical transformations. Currency is USD. The dataset is historical/static; the app refreshes the session, not live market data.')
