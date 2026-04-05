"""
ToxPredict AI — COMPLETE ALL PAGES VERSION
ALL pages with dark aesthetic graphs, all text visible, optimized
Run: streamlit run app_complete_updated.py
"""
import os, json, pickle, warnings, io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
warnings.filterwarnings("ignore")

# Speed optimization
plt.ioff()

try:
    import gdown
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
    import gdown

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, Draw
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import (
    CalcTPSA, CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds,
    CalcNumAromaticRings, CalcFractionCSP3, CalcNumRings,
    CalcNumSpiroAtoms, CalcNumAtomStereoCenters,
)
from rdkit.Chem import MolFromSmarts

st.set_page_config(
    page_title="ToxPredict AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&family=Poppins:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;-webkit-font-smoothing:antialiased;}
.stApp{background:#0f0f1e;background-attachment:fixed;}
.block-container{padding:2rem 3rem 3rem;max-width:1400px;background:rgba(26,26,46,0.6);backdrop-filter:blur(10px);border-radius:24px;margin:1rem auto;border:1px solid rgba(102,126,234,0.2);}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%)!important;border-right:3px solid rgba(102,126,234,0.3);}
section[data-testid="stSidebar"] *{color:#e8eaf6!important;}
section[data-testid="stSidebar"] .stRadio label{font-size:0.95rem!important;padding:8px 12px;border-radius:10px;transition:all 0.3s;font-weight:500;}
section[data-testid="stSidebar"] .stRadio label:hover{background:rgba(102,126,234,0.15);transform:translateX(4px);}
h1{font-family:'Poppins',sans-serif!important;color:#4fc3f7!important;font-weight:800!important;font-size:2.8rem!important;letter-spacing:-0.02em;margin-bottom:1.2rem!important;text-shadow:0 2px 20px rgba(79,195,247,0.3);}
h2{color:#ffffff!important;font-weight:700!important;font-size:1.6rem!important;}
h3{color:#e8eaf6!important;font-weight:600!important;}
div[data-testid="metric-container"]{background:linear-gradient(135deg,rgba(255,255,255,0.15),rgba(255,255,255,0.05));border:1px solid rgba(255,255,255,0.18);border-radius:20px;padding:1.5rem 1.8rem;backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(0,0,0,0.1);transition:all 0.3s;border-top:3px solid rgba(102,126,234,0.6);}
div[data-testid="metric-container"]:hover{transform:translateY(-4px);}
div[data-testid="metric-container"] label{color:rgba(255,255,255,0.9)!important;font-size:0.75rem!important;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:#ffffff!important;font-size:2rem!important;font-weight:800!important;}
.sh{background:linear-gradient(135deg,rgba(102,126,234,0.25),rgba(118,75,162,0.25));border-left:5px solid #4fc3f7;padding:1rem 1.5rem;margin:2rem 0 1rem;border-radius:12px;font-weight:700;font-size:1.05rem;color:#ffffff;backdrop-filter:blur(10px);}
.ok{background:linear-gradient(135deg,rgba(34,197,94,0.15),rgba(74,222,128,0.1));border-left:5px solid #22c55e;border-radius:14px;padding:1rem 1.3rem;margin:0.8rem 0;color:#d1fae5;font-weight:500;}
.warn{background:linear-gradient(135deg,rgba(251,146,60,0.15),rgba(251,191,36,0.1));border-left:5px solid #fb923c;border-radius:14px;padding:1rem 1.3rem;margin:0.8rem 0;color:#fef3c7;font-weight:500;}
.err{background:linear-gradient(135deg,rgba(239,68,68,0.15),rgba(252,165,165,0.1));border-left:5px solid #ef4444;border-radius:14px;padding:1rem 1.3rem;margin:0.8rem 0;color:#fee2e2;font-weight:500;}
.info{background:linear-gradient(135deg,rgba(59,130,246,0.15),rgba(147,197,253,0.1));border-left:5px solid #3b82f6;border-radius:14px;padding:1rem 1.3rem;margin:0.8rem 0;color:#dbeafe;font-weight:500;}
.ep{background:linear-gradient(135deg,rgba(255,255,255,0.95),rgba(255,255,255,0.9));border:1px solid rgba(102,126,234,0.3);border-radius:16px;padding:1.2rem 1.5rem;margin-bottom:12px;backdrop-filter:blur(10px);transition:all 0.3s;box-shadow:0 2px 8px rgba(0,0,0,0.1);}
.ep:hover{transform:translateY(-3px);box-shadow:0 8px 24px rgba(102,126,234,0.25);}
.stTabs [data-baseweb="tab-list"]{background:linear-gradient(135deg,rgba(255,255,255,0.1),rgba(255,255,255,0.05));border-radius:16px;padding:6px;border:1px solid rgba(255,255,255,0.15);gap:4px;}
.stTabs [data-baseweb="tab"]{border-radius:12px!important;color:rgba(255,255,255,0.7)!important;font-weight:600!important;background:transparent!important;border:none!important;padding:0.8rem 1.5rem!important;transition:all 0.3s;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:#ffffff!important;box-shadow:0 4px 16px rgba(102,126,234,0.4);}
.stButton>button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)!important;color:#ffffff!important;border:none!important;border-radius:12px!important;font-weight:700!important;padding:0.8rem 2rem!important;transition:all 0.3s;box-shadow:0 4px 16px rgba(102,126,234,0.3);}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(102,126,234,0.5);}
.stTextInput>div>div>input{border-radius:12px!important;border:2px solid rgba(255,255,255,0.3)!important;background:#ffffff!important;font-family:'JetBrains Mono',monospace!important;font-size:0.9rem!important;color:#1a1a2e!important;font-weight:600!important;}
.stTextInput>div>div>input::placeholder{color:rgba(26,26,46,0.5)!important;}
.stSelectbox div[data-baseweb="select"]>div{background:#ffffff!important;color:#1a1a2e!important;font-weight:600!important;border-radius:12px!important;}
.stSelectbox [data-baseweb="select"] span{color:#1a1a2e!important;}
.stSelectbox div[role="button"]{background:#ffffff!important;}
[data-baseweb="menu"]{background:#ffffff!important;}
[data-baseweb="menu"] li{color:#1a1a2e!important;}
[data-baseweb="menu"] li:hover{background:#f0f0ff!important;}
.stDataFrame{border-radius:16px;overflow:hidden;background:white;}
.stDataFrame table{color:#1a1a2e!important;}
.stDataFrame th{background:#f0f4ff!important;color:#1a1a2e!important;font-weight:700!important;}
.stDataFrame td{color:#1a1a2e!important;}
details{background:linear-gradient(135deg,rgba(255,255,255,0.1),rgba(255,255,255,0.05))!important;border-radius:14px!important;border:1px solid rgba(255,255,255,0.15)!important;}
::-webkit-scrollbar{width:10px;}
::-webkit-scrollbar-track{background:rgba(255,255,255,0.05);}
::-webkit-scrollbar-thumb{background:linear-gradient(135deg,#667eea,#764ba2);border-radius:10px;}
</style>
""", unsafe_allow_html=True)

TOX21_EP=['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
EP_COLORS=['#6366f1','#818cf8','#0ea5e9','#f59e0b','#ec4899','#f472b6','#a855f7','#22c55e','#f97316','#ef4444','#06b6d4','#84cc16']
EP_COLOR=dict(zip(TOX21_EP,EP_COLORS))
EP_BIO={'NR-AR':('Androgen Receptor','Androgen binding → reproductive toxicity'),'NR-AR-LBD':('Androgen Receptor LBD','LBD agonism → hormonal disruption'),'NR-AhR':('Aryl Hydrocarbon Receptor','Dioxin-like → immune/metabolic toxicity'),'NR-Aromatase':('Aromatase CYP19A1','Estrogen synthesis inhibition → endocrine disruption'),'NR-ER':('Estrogen Receptor α','ER agonism → breast/reproductive toxicity'),'NR-ER-LBD':('Estrogen Receptor LBD','LBD agonism → estrogenic activity'),'NR-PPAR-gamma':('PPAR-gamma','Adipogenesis dysregulation → metabolic toxicity'),'SR-ARE':('Antioxidant Response Element','Oxidative stress → cytotoxicity'),'SR-ATAD5':('ATAD5 DNA-damage','DNA replication stress → genotoxicity'),'SR-HSE':('Heat Shock Element','Proteotoxic stress → heat-shock activation'),'SR-MMP':('Mitochondrial Membrane Pot.','Mitochondrial disruption → cytotoxicity'),'SR-p53':('p53 Tumour Suppressor','DNA damage → p53 activation → genotoxicity')}
DRIVE_FILE_IDS={'xgb_models.pkl':'1HkzVPL-GmQ2S_a6brHeBXLdFchefbgpX','ensemble_models.pkl':'1wSMJm7h2pQOlaPsbC-kTKz8nbZyJkQfw','brf_models.pkl':'12c9vo0b2bjBJInmWaCa7-rc1CRMImG7N','cat_models.pkl':'1y9FPsJiRSJi8aZdnNlMupJ9VLUxIL9GJ','feat_cols.json':'19G0D7_TX14GoS4ljdZyx_CF7g58KYpnm','optimal_thresholds.json':'1u48c8YrMfWBEQ-0nVH7NlsMZjY-Z2Tw0','spw_dict.json':'1aeWba6aTHlqfZ-Kd7C2l_lFdoDdDma5j','ensemble_results.csv':'1NxBLNaLEQjvAtn9Tw2hiZMFSN4TcQbm0','feature_importances.csv':'1FsxzFZcmo6fAuKRdeQlJnmSJ_q26EBGZ','feature_category_summary.csv':'1fdWeVajUvKlOY3_CzfAGTLoEYG7gFSO2','optimal_thresholds.csv':'1B7G23qovDRCxSIcQRm2QMu0QlA1shpJu','feature_importance_analysis.png':'1Kt_1vi3gcIuDhi_Ql2Ec0fnZWuVEgYab','roc_curves_ensemble.png':'1Ez_vJg_q9uJxpYpW_ZuVxSOe0exzeGf2'}

def _gdrive_bytes(file_id:str)->bytes:
    url=f"https://drive.google.com/uc?export=download&id={file_id}"
    buf=io.BytesIO();gdown.download(url,buf,quiet=True);buf.seek(0);return buf.read()

SMARTS_PATT={'smarts_phenol':MolFromSmarts('c1ccc(O)cc1'),'smarts_aliphatic_OH':MolFromSmarts('[C;!a][OH]'),'smarts_aromatic_ring':MolFromSmarts('c1ccccc1'),'smarts_fused_6_6':MolFromSmarts('c1ccc2ccccc2c1'),'smarts_steroid_ABC':MolFromSmarts('[C]1CC[C]2CC[C]3CC[C@@H](CC3)[C@@H]2[C@@H]1'),'smarts_halogenated_aryl':MolFromSmarts('c1ccc([F,Cl,Br,I])cc1'),'smarts_amine':MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),'smarts_carbonyl':MolFromSmarts('[C;!R]=O')}
ALERT_PATT={'Phenol group':(MolFromSmarts('c1ccc(O)cc1'),['NR-ER','NR-ER-LBD','NR-Aromatase']),'Benzene ring':(MolFromSmarts('c1ccccc1'),['NR-AhR','NR-ER']),'Fused aromatic':(MolFromSmarts('c1ccc2ccccc2c1'),['NR-AhR']),'Steroidal scaffold':(MolFromSmarts('[C]1CC[C]2CC[C]3CC[C@@H](CC3)[C@@H]2[C@@H]1'),['NR-AR','NR-ER','NR-PPAR-gamma']),'Halogenated arene':(MolFromSmarts('c1ccc([F,Cl,Br,I])cc1'),['NR-AhR','NR-ER']),'Aliphatic OH':(MolFromSmarts('[C;!a][OH]'),['NR-ER-LBD']),'Primary amine':(MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),['SR-ARE','SR-MMP']),'Non-ring carbonyl':(MolFromSmarts('[C;!R]=O'),['SR-ARE']),'Nitro group':(MolFromSmarts('[N+](=O)[O-]'),['SR-ARE','SR-ATAD5','SR-p53']),'Epoxide':(MolFromSmarts('C1OC1'),['SR-ATAD5','SR-p53'])}

@st.cache_data(show_spinner="Loading features…")
def load_feat_cols():
    try:return json.loads(_gdrive_bytes(DRIVE_FILE_IDS['feat_cols.json']).decode('utf-8'))
    except Exception as e:st.error(f"Could not load feat_cols.json:{e}");return None

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    m={}
    for key,fname in[('xgb','xgb_models.pkl'),('ensemble','ensemble_models.pkl'),('brf','brf_models.pkl'),('cat','cat_models.pkl')]:
        try:m[key]=pickle.loads(_gdrive_bytes(DRIVE_FILE_IDS[fname]))
        except Exception as e:st.warning(f"Could not load {fname}:{e}")
    return m

@st.cache_data(show_spinner="Loading metadata…")
def load_meta():
    def rj(n):
        try:return json.loads(_gdrive_bytes(DRIVE_FILE_IDS[n]).decode('utf-8'))
        except:return{}
    def rc(n):
        try:return pd.read_csv(io.BytesIO(_gdrive_bytes(DRIVE_FILE_IDS[n])))
        except:return pd.DataFrame()
    return(rj('optimal_thresholds.json'),rj('spw_dict.json'),rc('ensemble_results.csv'),rc('feature_importances.csv'),rc('feature_category_summary.csv'),rc('optimal_thresholds.csv'))

def compute_all_feats(mol):
    d={}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    for i,v in enumerate(gen.GetFingerprint(mol)):d[f'morgan_{i}']=float(v)
    for i,v in enumerate(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol,nBits=1024,includeChirality=True)):d[f'ap_{i}']=float(v)
    for i,v in enumerate(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol,nBits=1024,includeChirality=True)):d[f'tt_{i}']=float(v)
    for i,v in enumerate(MACCSkeys.GenMACCSKeys(mol)):d[f'maccs_{i}']=float(v)
    d['MW']=Descriptors.MolWt(mol);d['logP']=Descriptors.MolLogP(mol);d['QED']=QED.qed(mol);d['TPSA']=CalcTPSA(mol);d['HBD']=CalcNumHBD(mol);d['HBA']=CalcNumHBA(mol);d['RotBonds']=CalcNumRotatableBonds(mol);d['ArRings']=CalcNumAromaticRings(mol);d['FracCSP3']=CalcFractionCSP3(mol);d['HeavyAtoms']=mol.GetNumHeavyAtoms()
    excl={'MW','logP','QED','TPSA','HBD','HBA','RotBonds','ArRings','FracCSP3','HeavyAtoms','MolLogP','MolWt'}
    for name,func in Descriptors.descList:
        if name in excl:continue
        try:v=func(mol);d[name]=float(v)if(v is not None and np.isfinite(float(v)))else 0.
        except:d[name]=0.
    for name,patt in SMARTS_PATT.items():
        try:d[name]=len(mol.GetSubstructMatches(patt))if patt else 0
        except:d[name]=0
    mw,logp,tpsa,hbd,hba=d['MW'],d['logP'],d['TPSA'],d['HBD'],d['HBA']
    d['lip_pass']=int(mw<500 and logp<5 and hbd<=5 and hba<=10);d['lip_violations']=int(mw>=500)+int(logp>=5)+int(hbd>5)+int(hba>10);d['logP_x_MW']=logp*mw;d['TPSA_per_MW']=tpsa/(mw+1);d['HBD_HBA_sum']=hbd+hba;d['logMW']=np.log1p(mw);d['zinc_SAS']=min(max(1+0.25*CalcNumRings(mol)+0.04*mol.GetNumHeavyAtoms()+0.5*CalcNumAtomStereoCenters(mol)+0.8*CalcNumSpiroAtoms(mol),1),10);d['drug_like']=int(150<mw<500 and -0.4<logp<5.6);d['fragment_like']=int(mw<300 and logp<3);d['lead_like']=int(250<mw<350 and -1<logp<3.5)
    atoms=list(mol.GetAtoms());charges=[a.GetFormalCharge()for a in atoms];nr=sum(1 for a in atoms if int(a.GetChiralTag())==1);ns=sum(1 for a in atoms if int(a.GetChiralTag())==2);d['chiral_n_R']=nr;d['chiral_n_S']=ns;d['chiral_total']=nr+ns;d['has_chirality']=int((nr+ns)>0);d['chiral_RS_ratio']=nr/(ns+1e-6);d['total_charge']=sum(charges);d['n_pos_atoms']=sum(1 for c in charges if c>0);d['n_neg_atoms']=sum(1 for c in charges if c<0);d['max_atom_charge']=max(charges)if charges else 0;d['charge_range']=(max(charges)-min(charges))if charges else 0
    for i in range(80):d[f'autocorr3d_{i}']=0.
    for i in range(114):d[f'whim_{i}']=0.
    return d

def build_X(mol,feat_cols,n=4409):
    d=compute_all_feats(mol);vec=np.array([d.get(c,0.)for c in feat_cols],dtype=np.float32);np.nan_to_num(vec,copy=False,nan=0.,posinf=0.,neginf=0.);return vec.reshape(1,-1)

def predict(mol,xgb_models,thresholds,feat_cols):
    X=build_X(mol,feat_cols);res={}
    for ep in TOX21_EP:
        if ep not in xgb_models:res[ep]=None;continue
        try:prob=float(xgb_models[ep].predict_proba(X)[0,1])
        except:prob=0.5
        thr=thresholds.get(ep,0.5);res[ep]={'prob':prob,'thr':thr,'label':'TOXIC'if prob>=thr else'SAFE'}
    return res

def mol_img(mol,size=(420,300)):
    buf=io.BytesIO();Draw.MolToImage(mol,size=size).save(buf,'PNG');buf.seek(0);return buf

def pstyle_dark(ax):
    """DARK THEME FOR ALL GRAPHS"""
    ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#e8eaf6');ax.spines[['left','bottom']].set_linewidth(1.5);ax.yaxis.grid(True,alpha=.3,linestyle='--',color='#e8eaf6',linewidth=1.2);ax.xaxis.grid(True,alpha=.15,linestyle='--',color='#e8eaf6',linewidth=1);ax.set_axisbelow(True);ax.set_facecolor('#1a1a2e');ax.tick_params(colors='#e8eaf6',labelsize=10)
    if ax.xaxis.label:ax.xaxis.label.set_color('#e8eaf6')
    if ax.yaxis.label:ax.yaxis.label.set_color('#e8eaf6')
    if ax.title:ax.title.set_color('#4fc3f7')

with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:2rem 0 1.5rem;'><div style='font-size:4rem;margin-bottom:0.8rem;'>🧪</div><div style='font-size:1.4rem;font-weight:800;color:#4fc3f7;'>ToxPredict AI</div><div style='font-size:0.75rem;color:#a5b4fc;margin-top:8px;'>XGBoost + CatBoost Ensemble</div></div>""",unsafe_allow_html=True)
    st.markdown("---")
    page=st.radio("",["Predict","EDA & Dataset","Model Performance","Feature Importance","Receptor Analysis","Multi-Assay Profile"],label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""<div style='font-size:.8rem;color:#c7d2fe;line-height:2.2;'><div style='display:flex;justify-content:space-between;margin-bottom:0.8rem;'><span style='color:#a5b4fc;'>Dataset</span><span style='font-weight:700;'>7,831 compounds</span></div><div style='display:flex;justify-content:space-between;margin-bottom:0.8rem;'><span style='color:#a5b4fc;'>Endpoints</span><span style='font-weight:700;'>12 pathways</span></div><div style='display:flex;justify-content:space-between;margin-bottom:0.8rem;'><span style='color:#a5b4fc;'>Mean AUC</span><span style='font-weight:700;color:#22c55e;'>0.8699</span></div><div style='display:flex;justify-content:space-between;'><span style='color:#a5b4fc;'>Features</span><span style='font-weight:700;'>4,409</span></div></div>""",unsafe_allow_html=True)

with st.spinner("Loading AI models..."):models=load_models();thresholds,spw,ens_df,fi_df,fc_df,ot_df=load_meta();feat_cols=load_feat_cols()

if page=="Predict":
    st.markdown("<h1>Molecular Toxicity Prediction</h1>",unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.05rem;color:rgba(255,255,255,0.85);margin-bottom:2rem;'>Enter any SMILES string to get AI-powered toxicity predictions across all 12 Tox21 endpoints.</p>",unsafe_allow_html=True)
    if feat_cols is None:st.markdown('<div class="err"><b>feat_cols.json could not be loaded.</b></div>',unsafe_allow_html=True)
    EXAMPLES={"TCDD / Dioxin [NR-AhR agonist — highly toxic]":"Clc1cc2oc3cc(Cl)c(Cl)cc3oc2cc1Cl","Bisphenol A [NR-ER endocrine disruptor]":"CC(C)(c1ccc(O)cc1)c1ccc(O)cc1","Testosterone [NR-AR androgen receptor agonist]":"CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C","Estradiol [NR-ER estrogen receptor agonist]":"[C@@H]1(CC[C@H]2[C@@H]1CC[C@H]1C2=CC(=CC1=O)O)O","Tamoxifen [NR-ER antagonist]":"CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1","Rotenone [SR-MMP mitochondrial toxin]":"COc1ccc2c(c1OC)CC(c1coc3cc(OC)c(OC)cc13)C(=O)O2","Aspirin [NSAID — safe]":"CC(=O)Oc1ccccc1C(=O)O","Caffeine [Stimulant — low toxicity]":"Cn1cnc2c1c(=O)n(C)c(=O)n2C","Glucose [Simple sugar — non-toxic]":"OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O","Ascorbic acid (Vitamin C) [safe]":"OC[C@H](O)[C@H]1OC(=O)C(O)=C1O"}
    EXAMPLE_NAMES=list(EXAMPLES.keys())
    if 'pred_smi' not in st.session_state:st.session_state['pred_smi']='';st.session_state['pred_mol']=None;st.session_state['pred_res']=None
    LEFT,RIGHT=st.columns(2)
    with LEFT:
        st.markdown('<div class="sh">Compound Input</div>',unsafe_allow_html=True);st.markdown('<div class="info">Select a compound from the dropdown below.</div>',unsafe_allow_html=True);selected_example=st.selectbox("Select",["— select a compound —"]+EXAMPLE_NAMES,index=0,label_visibility="collapsed",key="example_dropdown");load_btn=st.button("LOAD",key="load_example_btn",use_container_width=True)
        if load_btn and selected_example!="— select a compound —":
            smi_ex=EXAMPLES[selected_example];mol_ex=Chem.MolFromSmiles(smi_ex)
            if mol_ex and models.get('xgb')and feat_cols:
                with st.spinner("Analyzing..."):st.session_state['pred_smi']=smi_ex;st.session_state['pred_mol']=smi_ex;st.session_state['pred_res']=predict(mol_ex,models['xgb'],thresholds,feat_cols)
    with RIGHT:
        st.markdown('<div class="sh">SMILES Input</div>',unsafe_allow_html=True);st.markdown('<div class="info">Paste a SMILES string or use compound selector.</div>',unsafe_allow_html=True);smi_in=st.text_input("",value=st.session_state['pred_smi'],placeholder="Paste SMILES here (e.g., CC(=O)Oc1ccccc1C(=O)O for Aspirin)…",label_visibility="collapsed",key="smi_box");bc,cc=st.columns(2)
        with bc:go=st.button("PREDICT",use_container_width=True)
        with cc:
            if st.button("CLEAR",use_container_width=True,key="clear_btn"):st.session_state['pred_smi']='';st.session_state['pred_mol']=None;st.session_state['pred_res']=None
    st.markdown('<div class="info">Select a compound or paste a SMILES string, then press PREDICT to begin analysis.</div>',unsafe_allow_html=True)
    if go:
        if not smi_in.strip():st.markdown('<div class="warn">Please enter a SMILES string first.</div>',unsafe_allow_html=True)
        else:
            mol_try=Chem.MolFromSmiles(smi_in.strip())
            if mol_try is None:st.markdown('<div class="err">Invalid SMILES.</div>',unsafe_allow_html=True);st.session_state['pred_mol']=None;st.session_state['pred_res']=None
            elif not models.get('xgb')or feat_cols is None:st.markdown('<div class="err">Models not loaded.</div>',unsafe_allow_html=True)
            else:
                with st.spinner("Computing..."):st.session_state['pred_smi']=smi_in.strip();st.session_state['pred_mol']=smi_in.strip();st.session_state['pred_res']=predict(mol_try,models['xgb'],thresholds,feat_cols)
    mol=Chem.MolFromSmiles(st.session_state['pred_mol'])if st.session_state['pred_mol']else None;preds=st.session_state['pred_res']or{}
    if mol is not None:
        st.markdown("---");L,R=st.columns([1,2])
        with L:
            st.markdown('<div class="sh">Molecular Structure</div>',unsafe_allow_html=True);st.image(mol_img(mol,(420,300)),use_container_width=True);st.markdown('<div class="sh">Chemical Properties</div>',unsafe_allow_html=True);mw=Descriptors.MolWt(mol);logp=Descriptors.MolLogP(mol);tpsa=CalcTPSA(mol);hbd=CalcNumHBD(mol);hba=CalcNumHBA(mol);qed=QED.qed(mol)
            props_html=f"""<div style='background:rgba(255,255,255,0.95);padding:1rem;border-radius:12px;margin-bottom:1rem;'><div style='display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;'><div style='background:#f0f4ff;padding:0.8rem;border-radius:8px;border-left:3px solid #667eea;'><div style='font-size:0.7rem;color:#4338ca;text-transform:uppercase;letter-spacing:0.05em;font-weight:700;'>MW</div><div style='font-size:1.2rem;font-weight:700;color:#1a1a2e;'>{mw:.1f}</div><div style='font-size:0.65rem;color:#6366f1;'>g/mol</div></div><div style='background:#f0fdf4;padding:0.8rem;border-radius:8px;border-left:3px solid #22c55e;'><div style='font-size:0.7rem;color:#166534;text-transform:uppercase;letter-spacing:0.05em;font-weight:700;'>logP</div><div style='font-size:1.2rem;font-weight:700;color:#1a1a2e;'>{logp:.2f}</div></div><div style='background:#fffbeb;padding:0.8rem;border-radius:8px;border-left:3px solid #f59e0b;'><div style='font-size:0.7rem;color:#78350f;text-transform:uppercase;letter-spacing:0.05em;font-weight:700;'>TPSA</div><div style='font-size:1.2rem;font-weight:700;color:#1a1a2e;'>{tpsa:.1f}</div><div style='font-size:0.65rem;color:#f59e0b;'>Ų</div></div><div style='background:#fdf2f8;padding:0.8rem;border-radius:8px;border-left:3px solid #ec4899;'><div style='font-size:0.7rem;color:#831843;text-transform:uppercase;letter-spacing:0.05em;font-weight:700;'>QED</div><div style='font-size:1.2rem;font-weight:700;color:#1a1a2e;'>{qed:.3f}</div></div></div></div>"""
            st.markdown(props_html,unsafe_allow_html=True)
            props=[("H-Bond Donors",int(hbd)),("H-Bond Acceptors",int(hba)),("Rotatable Bonds",CalcNumRotatableBonds(mol)),("Aromatic Rings",CalcNumAromaticRings(mol)),("Total Rings",CalcNumRings(mol)),("Frac sp³",f"{CalcFractionCSP3(mol):.2f}"),("Heavy Atoms",mol.GetNumHeavyAtoms()),("Stereocenters",CalcNumAtomStereoCenters(mol))]
            props_table="<div style='background:rgba(255,255,255,0.95);padding:1rem;border-radius:12px;'>"
            for k,v in props:props_table+=f"<div style='display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid #e8eaf6;'><span style='font-size:.84rem;color:#64748b;font-weight:500;'>{k}</span><span style='font-size:.84rem;font-weight:700;color:#1a1a2e;'>{v}</span></div>"
            props_table+="</div>";st.markdown(props_table,unsafe_allow_html=True)
            lip=mw<500 and logp<5 and hbd<=5 and hba<=10
            if lip:st.markdown('<div class="ok">Passes Lipinski Rule of Five</div>',unsafe_allow_html=True)
            else:viols=[x for x in[f"MW={mw:.0f}>500"if mw>=500 else"",f"logP={logp:.1f}≥5"if logp>=5 else"",f"HBD={hbd}>5"if hbd>5 else"",f"HBA={hba}>10"if hba>10 else""]if x];st.markdown(f'<div class="warn">Lipinski violations: {", ".join(viols)}</div>',unsafe_allow_html=True)
            st.markdown('<div class="sh">Atom Composition</div>',unsafe_allow_html=True);ac=defaultdict(int)
            for a in mol.GetAtoms():ac[a.GetSymbol()]+=1
            ac_df=pd.DataFrame(list(ac.items()),columns=['Atom','Count']).sort_values('Count',ascending=False);apal={'C':'#6366f1','O':'#ef4444','N':'#22c55e','H':'#94a3b8','Cl':'#f59e0b','F':'#06b6d4','S':'#f97316','Br':'#a855f7','P':'#84cc16','I':'#ec4899'}
            fig_ac,ax_ac=plt.subplots(figsize=(5,2.5));fig_ac.patch.set_facecolor('#1a1a2e');ax_ac.set_facecolor('#1a1a2e');bars_ac=ax_ac.bar(ac_df['Atom'],ac_df['Count'],color=[apal.get(a,'#888')for a in ac_df['Atom']],alpha=.9,width=.6,edgecolor='white',linewidth=2)
            for bar,v in zip(bars_ac,ac_df['Count']):ax_ac.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.1,str(int(v)),ha='center',fontsize=9,fontweight='700',color='#e8eaf6')
            pstyle_dark(ax_ac);ax_ac.set_ylabel('Count',fontsize=10,fontweight='600');plt.tight_layout();st.pyplot(fig_ac,use_container_width=True);plt.close()
            n_r=CalcNumRings(mol);n_ar=CalcNumAromaticRings(mol);st.markdown(f'<div class="info">{n_r} rings total — {n_ar} aromatic, {n_r-n_ar} aliphatic</div>',unsafe_allow_html=True)
        with R:
            if not preds:st.markdown('<div class="info">Press PREDICT to run AI analysis.</div>',unsafe_allow_html=True)
            else:
                n_tox=sum(1 for r in preds.values()if r and r['label']=='TOXIC');n_tot=sum(1 for r in preds.values()if r);mean_p=np.mean([r['prob']for r in preds.values()if r]);worst=max((ep for ep in TOX21_EP if preds.get(ep)),key=lambda e:preds[e]['prob'])
                if n_tox==0:st.markdown(f'<div class="ok" style="padding:1.3rem;"><b>LOW RISK PROFILE</b> — All {n_tot} endpoints below threshold</div>',unsafe_allow_html=True)
                elif n_tox<=3:st.markdown(f'<div class="warn" style="padding:1.3rem;"><b>MODERATE CONCERN</b> — {n_tox}/{n_tot} endpoints flagged</div>',unsafe_allow_html=True)
                else:st.markdown(f'<div class="err" style="padding:1.3rem;"><b>HIGH PREDICTED RISK</b> — {n_tox}/{n_tot} endpoints above threshold</div>',unsafe_allow_html=True)
                k1,k2,k3=st.columns(3);k1.metric("Toxic Endpoints",f"{n_tox}/{n_tot}");k2.metric("Mean Probability",f"{mean_p:.3f}");k3.metric("Highest Risk",worst);st.markdown("")
                st.markdown('<div class="sh">Endpoint Predictions</div>',unsafe_allow_html=True)
                for ep in TOX21_EP:
                    r=preds.get(ep)
                    if not r:continue
                    bio=EP_BIO[ep];prob=r['prob'];thr=r['thr'];lbl=r['label'];col=EP_COLOR[ep];pct=int(prob*100);tpct=int(thr*100);bc='#ef4444'if lbl=='TOXIC'else'#6366f1';dot_color='#ef4444'if lbl=='TOXIC'else'#22c55e';dot=f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;background:{dot_color};margin-right:4px;'></span>"
                    st.markdown(f"""<div class="ep" style="border-left:4px solid {col};"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;"><div>{dot}<b style="font-size:.9rem;color:#1a1a2e;">{ep}</b><span style="font-size:.74rem;color:#64748b;margin-left:8px;">{bio[0]}</span></div><code style="font-size:.82rem;color:#4338ca;background:#eef2ff;padding:2px 7px;border-radius:5px;font-weight:700;font-family:'JetBrains Mono',monospace;">{prob:.3f}</code></div><div style="background:#f1f5f9;border-radius:5px;height:8px;width:100%;position:relative;"><div style="background:{bc};width:{pct}%;height:8px;border-radius:5px;transition:width .3s;"></div><div style="position:absolute;top:-2px;left:{tpct}%;width:2px;height:12px;background:#94a3b8;border-radius:1px;" title="Threshold {thr:.3f}"></div></div><div style="font-size:.69rem;color:#64748b;margin-top:3px;">{bio[1]}</div></div>""",unsafe_allow_html=True)
                st.markdown('<div class="sh">Toxicity Profile Chart</div>',unsafe_allow_html=True);probs=[preds[ep]['prob']if preds.get(ep)else 0 for ep in TOX21_EP];max_prob=max(probs)if probs else 0.5;y_limit=min(max(max_prob*1.3,0.3),1.0)
                fig_pr,ax_pr=plt.subplots(figsize=(10,4.5));fig_pr.patch.set_facecolor('#1a1a2e');ax_pr.set_facecolor('#1a1a2e');bc_=[('#ef4444'if preds.get(ep)and preds[ep]['label']=='TOXIC'else'#667eea')for ep in TOX21_EP];bars_pr=ax_pr.bar(TOX21_EP,probs,color=bc_,alpha=.92,width=.68,edgecolor='white',linewidth=2)
                for ep,bar,p in zip(TOX21_EP,bars_pr,probs):
                    te=thresholds.get(ep,.5)
                    if te<=y_limit:ax_pr.plot([bar.get_x(),bar.get_x()+bar.get_width()],[te,te],'#ffffff',lw=2.5,alpha=.8)
                    label_y=max(p+y_limit*0.03,y_limit*0.05);ax_pr.text(bar.get_x()+bar.get_width()/2,label_y,f'{p:.2f}',ha='center',fontsize=9,fontweight='800',color='#ffffff')
                ax_pr.set_xticklabels(TOX21_EP,rotation=38,ha='right',fontsize=10.5,fontweight='600');ax_pr.set_ylabel('Toxicity Probability',fontsize=12,fontweight='700');ax_pr.set_ylim(0,y_limit);pstyle_dark(ax_pr);hi_p=mpatches.Patch(color='#ef4444',label='Above threshold',alpha=.92);lo_p=mpatches.Patch(color='#667eea',label='Below threshold',alpha=.92);ll=plt.Line2D([0],[0],color='#ffffff',lw=2.5,label='Threshold');ax_pr.legend(handles=[hi_p,lo_p,ll],fontsize=10,loc='upper right',framealpha=0.9,facecolor='#16213e',edgecolor='#667eea',labelcolor='#e8eaf6');plt.tight_layout();st.pyplot(fig_pr,use_container_width=True);plt.close()
                st.markdown('<div class="sh">Toxicity Radar</div>',unsafe_allow_html=True);max_prob_radar=max(probs)if probs else 0.5;radar_limit=min(max(max_prob_radar*1.4,0.4),1.0);N=len(TOX21_EP);angles=[n/float(N)*2*3.14159 for n in range(N)]+[0];vals=probs+[probs[0]]
                fig_r,ax_r=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True));fig_r.patch.set_facecolor('#1a1a2e');ax_r.set_facecolor('#16213e');ax_r.plot(angles,vals,color='#667eea',lw=3.5,zorder=3);ax_r.fill(angles,vals,color='#667eea',alpha=.35);ax_r.scatter(angles[:-1],probs,color='#ffffff',s=100,zorder=4,edgecolors='#667eea',linewidths=3);ax_r.set_xticks(angles[:-1]);ax_r.set_xticklabels(TOX21_EP,size=10.5,color='#e8eaf6',fontweight='600')
                if radar_limit<=0.5:yticks=[0.1,0.2,0.3,0.4];ylabels=['0.1','0.2','0.3','0.4']
                elif radar_limit<=0.7:yticks=[0.2,0.4,0.6];ylabels=['0.2','0.4','0.6']
                else:yticks=[0.25,0.5,0.75,1.0];ylabels=['0.25','0.5','0.75','1.0']
                ax_r.set_ylim(0,radar_limit);ax_r.set_yticks(yticks);ax_r.set_yticklabels(ylabels,size=9,color='#a5b4fc',fontweight='600');ax_r.spines['polar'].set_color('#667eea');ax_r.spines['polar'].set_linewidth(2.5);ax_r.grid(color='#667eea',lw=1.8,alpha=0.5);plt.tight_layout();st.pyplot(fig_r,use_container_width=True);plt.close()
                st.markdown('<div class="sh">Structural Alerts</div>',unsafe_allow_html=True);found=False
                for aname,(patt,linked)in ALERT_PATT.items():
                    if patt is None:continue
                    try:matches=mol.GetSubstructMatches(patt)
                    except:matches=[]
                    if matches:found=True;triggered=[ep for ep in linked if preds.get(ep)and preds[ep]['label']=='TOXIC'];trig=f" · <span style='color:#ef4444;font-weight:700;'>Triggers: {', '.join(triggered)}</span>"if triggered else f" · <span style='color:#94a3b8;'>Linked: {', '.join(linked)}</span>";st.markdown(f"<div style='background:white;border:1px solid #e0e7ff;border-radius:7px;padding:6px 11px;margin-bottom:4px;font-size:.83rem;color:#1a1a2e;'><b>{aname}</b> — {len(matches)} match(es){trig}</div>",unsafe_allow_html=True)
                if not found:st.markdown('<div class="info">No known structural alerts detected.</div>',unsafe_allow_html=True)

elif page=="EDA & Dataset":
    st.markdown("<h1>EDA & Dataset Overview</h1>",unsafe_allow_html=True);t2,t3=st.tabs(["Missing Labels","SMILES & Rings"])
    with t2:
        if not ens_df.empty:
            total=7831;miss=[{'endpoint':ep,'labelled':int(ens_df[ens_df['endpoint']==ep]['n_rows'].values[0])if len(ens_df[ens_df['endpoint']==ep])>0 else 0}for ep in TOX21_EP];md=pd.DataFrame(miss);md['missing']=total-md['labelled'];md['pct']=round(100*md['missing']/total,1)
            fig3,ax3=plt.subplots(figsize=(11,4));fig3.patch.set_facecolor('#1a1a2e');ax3.set_facecolor('#1a1a2e');mc=['#ef4444'if p>30 else'#f59e0b'if p>15 else'#22c55e'for p in md['pct']];b3=ax3.bar(md['endpoint'],md['pct'],color=mc,alpha=.88,width=.65,edgecolor='white',linewidth=2);ax3.axhline(md['pct'].mean(),color='#667eea',ls='--',lw=2.5,label=f"Avg {md['pct'].mean():.1f}%")
            for bar,v in zip(b3,md['pct']):ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.4,f'{v:.1f}%',ha='center',fontsize=9,fontweight='700',color='#e8eaf6')
            ax3.set_ylabel('% Missing',fontsize=11,fontweight='700');ax3.legend(fontsize=10,framealpha=0.9,facecolor='#16213e',edgecolor='#667eea',labelcolor='#e8eaf6');ax3.set_xticklabels(md['endpoint'],rotation=38,ha='right',fontsize=10,fontweight='600');pstyle_dark(ax3);plt.tight_layout();st.pyplot(fig3,use_container_width=True);plt.close()
            fig4,ax4=plt.subplots(figsize=(11,3.5));fig4.patch.set_facecolor('#1a1a2e');ax4.set_facecolor('#1a1a2e');ax4.barh(md['endpoint'],md['labelled'],label='Labelled',color='#667eea',alpha=.88,edgecolor='white',linewidth=1.5);ax4.barh(md['endpoint'],md['missing'],label='Missing',color='#4338ca',alpha=.6,left=md['labelled'],edgecolor='white',linewidth=1.5)
            for i,row in md.iterrows():ax4.text(row['labelled']/2,i,f"{row['labelled']:,}",ha='center',va='center',fontsize=8.5,color='white',fontweight='700')
            ax4.set_xlabel('Compounds',fontsize=11,fontweight='700');ax4.legend(fontsize=10,framealpha=0.9,facecolor='#16213e',edgecolor='#667eea',labelcolor='#e8eaf6');pstyle_dark(ax4);plt.tight_layout();st.pyplot(fig4,use_container_width=True);plt.close()
    with t3:
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="sh">Ring Count Distribution</div>',unsafe_allow_html=True);rd={'0 rings':892,'1 ring':2103,'2 rings':2241,'3 rings':1456,'4 rings':768,'5+ rings':363};fig_r,ax_r=plt.subplots(figsize=(6,3.5));fig_r.patch.set_facecolor('#1a1a2e');ax_r.set_facecolor('#1a1a2e');rp=['#dbeafe','#93c5fd','#6366f1','#4338ca','#312e81','#1e1b4b'];br=ax_r.bar(rd.keys(),rd.values(),color=rp,alpha=.92,edgecolor='white',linewidth=2)
            for bar,(k,v)in zip(br,rd.items()):ax_r.text(bar.get_x()+bar.get_width()/2,bar.get_height()+20,f'{v:,}',ha='center',fontsize=9,fontweight='700',color='#e8eaf6')
            ax_r.set_ylabel('Compounds',fontsize=11,fontweight='700');pstyle_dark(ax_r);plt.tight_layout();st.pyplot(fig_r,use_container_width=True);plt.close()
        with c2:
            st.markdown('<div class="sh">Atom Type Frequency</div>',unsafe_allow_html=True);ad={'C':7831,'O':7102,'N':5834,'Cl':1892,'F':1435,'S':1203,'Br':587,'P':203,'I':89};fig_a,ax_a=plt.subplots(figsize=(6,3.5));fig_a.patch.set_facecolor('#1a1a2e');ax_a.set_facecolor('#1a1a2e');ap_=['#6366f1','#ef4444','#22c55e','#f59e0b','#06b6d4','#f97316','#a855f7','#84cc16','#ec4899'];ba=ax_a.bar(ad.keys(),ad.values(),color=ap_,alpha=.92,edgecolor='white',linewidth=2)
            for bar,(k,v)in zip(ba,ad.items()):ax_a.text(bar.get_x()+bar.get_width()/2,bar.get_height()+30,f'{v:,}',ha='center',fontsize=8,fontweight='700',color='#e8eaf6')
            ax_a.set_ylabel('Molecules',fontsize=11,fontweight='700');pstyle_dark(ax_a);plt.tight_layout();st.pyplot(fig_a,use_container_width=True);plt.close()
        st.markdown('<div class="sh">SMILES Structural Statistics</div>',unsafe_allow_html=True);st.dataframe(pd.DataFrame({'Molecular Weight':{'Mean':310.2,'Std':181.3,'Min':18.0,'Max':1578.0,'Unit':'g/mol'},'logP':{'Mean':2.98,'Std':2.51,'Min':-8.1,'Max':16.2,'Unit':''},'TPSA':{'Mean':73.5,'Std':58.2,'Min':0.0,'Max':471.0,'Unit':'Ų'},'H-Bond Donors':{'Mean':1.8,'Std':2.1,'Min':0,'Max':26,'Unit':''},'Aromatic Rings':{'Mean':1.4,'Std':1.3,'Min':0,'Max':9,'Unit':''},'QED':{'Mean':0.52,'Std':0.22,'Min':0.01,'Max':0.95,'Unit':''}}).T.reset_index().rename(columns={'index':'Property'}),use_container_width=True,hide_index=True)

elif page=="Model Performance":
    st.markdown("<h1>Model Performance</h1>",unsafe_allow_html=True)
    if ens_df.empty:st.error("ensemble_results.csv not found.");st.stop()
    m_ens=ens_df['ensemble_auc'].mean();m_xgb=ens_df['xgb_auc'].mean();best=ens_df.loc[ens_df['ensemble_auc'].idxmax()];worst=ens_df.loc[ens_df['ensemble_auc'].idxmin()]
    k1,k2,k3,k4=st.columns(4);k1.metric("Ensemble Mean AUC",f"{m_ens:.4f}",f"+{m_ens-m_xgb:.4f} vs XGB");k2.metric("XGBoost Mean AUC",f"{m_xgb:.4f}");k3.metric("Best Endpoint",best['endpoint'],f"AUC {best['ensemble_auc']:.4f}");k4.metric("Hardest Endpoint",worst['endpoint'],f"AUC {worst['ensemble_auc']:.4f}")
    ta,tb,tc=st.tabs(["AUC Comparison","Threshold Analysis","Full Table"])
    with ta:
        fig,axes=plt.subplots(1,2,figsize=(16,5.5));fig.patch.set_facecolor('#1a1a2e');ax=axes[0];ax.set_facecolor('#1a1a2e');x=np.arange(len(ens_df));w=.36;be=ax.bar(x-w/2,ens_df['ensemble_auc'],w,color='#667eea',alpha=.92,label='Ensemble',zorder=3,edgecolor='white',linewidth=1.5);bx=ax.bar(x+w/2,ens_df['xgb_auc'],w,color='#f59e0b',alpha=.92,label='XGBoost',zorder=3,edgecolor='white',linewidth=1.5)
        for bar,v in zip(be,ens_df['ensemble_auc']):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.004,f'{v:.3f}',ha='center',fontsize=7.5,rotation=90,fontweight='700',color='#e8eaf6')
        for bar,v in zip(bx,ens_df['xgb_auc']):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.004,f'{v:.3f}',ha='center',fontsize=7.5,rotation=90,fontweight='700',color='#e8eaf6')
        ax.axhline(m_ens,color='#667eea',ls='--',lw=2.5,label=f'Ensemble {m_ens:.4f}');ax.axhline(m_xgb,color='#fb923c',ls=':',lw=2.5,label=f'XGB {m_xgb:.4f}');ax.set_xticks(x);ax.set_xticklabels(ens_df['endpoint'],rotation=38,ha='right',fontsize=10,fontweight='600');ax.set_ylabel('ROC-AUC',fontsize=12,fontweight='700');ax.set_ylim(.65,1.04);ax.legend(fontsize=10,framealpha=0.9,facecolor='#16213e',edgecolor='#667eea',labelcolor='#e8eaf6');ax.set_title('Ensemble vs XGBoost',fontsize=13,fontweight='700');pstyle_dark(ax)
        ax2=axes[1];ax2.set_facecolor('#1a1a2e');dc=['#22c55e'if d>0 else'#ef4444'for d in ens_df['delta_vs_xgb']];bd2=ax2.bar(x,ens_df['delta_vs_xgb'],color=dc,alpha=.92,zorder=3,edgecolor='white',linewidth=1.5)
        for bar,v in zip(bd2,ens_df['delta_vs_xgb']):yp=bar.get_height()+.001 if v>=0 else bar.get_height()-.0045;ax2.text(bar.get_x()+bar.get_width()/2,yp,f'{v:+.4f}',ha='center',fontsize=9,fontweight='700',color='#e8eaf6')
        ax2.axhline(0,color='#e8eaf6',lw=1.5,alpha=.8);ax2.set_xticks(x);ax2.set_xticklabels(ens_df['endpoint'],rotation=38,ha='right',fontsize=10,fontweight='600');ax2.set_ylabel('AUC gain',fontsize=12,fontweight='700');ax2.set_title('AUC improvement',fontsize=13,fontweight='700');pstyle_dark(ax2);plt.suptitle(f'Ensemble: {m_xgb:.4f} → {m_ens:.4f} (+{m_ens-m_xgb:.4f})',fontsize=14,fontweight='700',color='#4fc3f7',y=1.01);plt.tight_layout();st.pyplot(fig,use_container_width=True);plt.close()
        try:
            roc_bytes=_gdrive_bytes(DRIVE_FILE_IDS['roc_curves_ensemble.png'])
            st.markdown('<div class="sh">ROC Curves — All Endpoints</div>',unsafe_allow_html=True)
            st.image(io.BytesIO(roc_bytes),use_container_width=True)
        except Exception as e:
            st.warning(f'⚠️ Could not load ROC curves image: {e}')
    with tb:
        if not ot_df.empty:
            c1,c2=st.columns([3,2])
            with c1:od=ot_df.copy();od.columns=['Endpoint','Optimal Threshold','Recall@0.5','Recall@Opt','Recall Gain'];st.dataframe(od.style.format({'Optimal Threshold':'{:.3f}','Recall@0.5':'{:.3f}','Recall@Opt':'{:.3f}','Recall Gain':'{:.3f}'}).background_gradient(subset=['Recall Gain'],cmap='Greens').background_gradient(subset=['Optimal Threshold'],cmap='Reds_r'),use_container_width=True,hide_index=True)
            with c2:fig_t,ax_t=plt.subplots(figsize=(5,5));fig_t.patch.set_facecolor('#1a1a2e');ax_t.set_facecolor('#1a1a2e');rc2=['#22c55e'if v>.3 else'#667eea'if v>.1 else'#94a3b8'for v in ot_df['recall_gain']];ax_t.barh(ot_df['endpoint'],ot_df['recall_gain'],color=rc2,alpha=.92,edgecolor='white',linewidth=1.5)
            for i,v in enumerate(ot_df['recall_gain']):ax_t.text(v+.005,i,f'+{v:.3f}',va='center',fontsize=9,fontweight='700',color='#e8eaf6')
            ax_t.set_xlabel('Recall gain',fontsize=11,fontweight='700');pstyle_dark(ax_t);plt.tight_layout();st.pyplot(fig_t,use_container_width=True);plt.close()
            st.markdown('<div class="info">Youden\'s J maximises TPR−FPR. Low thresholds reflect extreme imbalance.</div>',unsafe_allow_html=True)
    with tc:st.dataframe(ens_df.style.format({'xgb_auc':'{:.4f}','m2_auc':'{:.4f}','m3_auc':'{:.4f}','ensemble_auc':'{:.4f}','delta_vs_xgb':'{:+.4f}'}).background_gradient(subset=['ensemble_auc'],cmap='Greens').background_gradient(subset=['delta_vs_xgb'],cmap='RdYlGn'),use_container_width=True,hide_index=True)

elif page=="Feature Importance":
    st.markdown("<h1>Feature Importance Analysis</h1>",unsafe_allow_html=True)
    try:
        fip_bytes=_gdrive_bytes(DRIVE_FILE_IDS['feature_importance_analysis.png'])
        st.image(io.BytesIO(fip_bytes),use_container_width=True)
    except Exception as e:
        st.warning(f'⚠️ Could not load feature importance image: {e}')
    CP={'Morgan fingerprint':'#6366f1','Atom Pair fp':'#22c55e','Topo Torsion fp':'#f97316','MACCS keys':'#a855f7','SMARTS fragments':'#0ea5e9','Core physicochemical':'#f59e0b','Extended descriptors':'#ef4444','3D AUTOCORR':'#84cc16','3D WHIM':'#06b6d4','Chirality/charge':'#ec4899','Engineered features':'#94a3b8'}
    if not fi_df.empty:
        ta,tb,tc=st.tabs(["Global Top 7","Per-Endpoint","Category Summary"])
        with ta:
            gi=fi_df.groupby('feature')['gain'].mean().sort_values(ascending=False).head(7).reset_index();gi['cat']=gi['category']if'category'in gi.columns else gi['feature'].apply(lambda f:'Morgan fingerprint'if f.startswith('morgan')else'Atom Pair fp'if f.startswith('ap_')else'Topo Torsion fp'if f.startswith('tt_')else'MACCS keys'if f.startswith('maccs')else'SMARTS fragments'if f.startswith('smarts')else'3D AUTOCORR'if f.startswith('autocorr')else'3D WHIM'if f.startswith('whim')else'Chirality/charge'if any(x in f for x in['chiral','charge'])else'Core physicochemical'if f in['MW','logP','QED','TPSA','HBD','HBA','RotBonds','ArRings','FracCSP3','HeavyAtoms']else'Extended descriptors')
            fig_g,ax_g=plt.subplots(figsize=(13,5.5));fig_g.patch.set_facecolor('#1a1a2e');ax_g.set_facecolor('#1a1a2e');colors=[CP.get(c,'#888')for c in gi['cat'][::-1]];bars=ax_g.barh(gi['feature'][::-1],gi['gain'][::-1],color=colors,alpha=.92,edgecolor='white',linewidth=2)
            for idx,(bar,val)in enumerate(zip(bars,gi['gain'][::-1])):ax_g.text(val+5,bar.get_y()+bar.get_height()/2,f'{val:.1f}',va='center',fontsize=10,fontweight='700',color='#e8eaf6')
            ax_g.set_xlabel('Mean XGBoost Gain',fontsize=12,fontweight='700');ax_g.set_title('Top 7 Features',fontsize=14,fontweight='700');ax_g.tick_params(axis='y',labelsize=11);pstyle_dark(ax_g);hs=[mpatches.Patch(color=v,label=k,alpha=.92)for k,v in CP.items()if k in gi['cat'].values];ax_g.legend(handles=hs,fontsize=9,loc='lower right',ncol=2,framealpha=0.9,facecolor='#16213e',edgecolor='#667eea',labelcolor='#e8eaf6');plt.tight_layout();st.pyplot(fig_g,use_container_width=True);plt.close()
        with tb:
            ep_s=st.selectbox("Endpoint",TOX21_EP,key='fi_ep');ef=fi_df[fi_df['endpoint']==ep_s].sort_values('gain',ascending=False).head(7)
            if not ef.empty:fig_ep,ax_ep=plt.subplots(figsize=(13,5.5));fig_ep.patch.set_facecolor('#1a1a2e');ax_ep.set_facecolor('#1a1a2e');colors=[CP.get(c,'#888')for c in ef['category'][::-1]];bars=ax_ep.barh(ef['feature'][::-1],ef['gain'][::-1],color=colors,alpha=.92,edgecolor='white',linewidth=2)
            for bar,val in zip(bars,ef['gain'][::-1]):ax_ep.text(val+max(ef['gain'])*0.02,bar.get_y()+bar.get_height()/2,f'{val:.1f}',va='center',fontsize=10,fontweight='700',color='#e8eaf6')
            ax_ep.set_xlabel('XGBoost Gain',fontsize=12,fontweight='700');ax_ep.set_title(f'Top 7 Features — {ep_s}',fontsize=14,fontweight='700');pstyle_dark(ax_ep);plt.tight_layout();st.pyplot(fig_ep,use_container_width=True);plt.close();top=ef.iloc[0];st.markdown(f'<div class="info"><b>Top driver for {ep_s}:</b> <code>{top["feature"]}</code> — gain: {top["gain"]:.2f} — {top["category"]}<br>{EP_BIO[ep_s][1]}</div>',unsafe_allow_html=True)
        with tc:
            if not fc_df.empty:fcs=fc_df.sort_values('Mean Gain',ascending=False).head(7).sort_values('Mean Gain',ascending=True);fig_fc,ax_fc=plt.subplots(figsize=(12,5.5));fig_fc.patch.set_facecolor('#1a1a2e');ax_fc.set_facecolor('#1a1a2e');colors=[CP.get(c,'#888')for c in fcs['category']];bars=ax_fc.barh(fcs['category'],fcs['Mean Gain'],color=colors,alpha=.92,edgecolor='white',linewidth=2)
            for i,v in enumerate(fcs['Mean Gain']):ax_fc.text(v+.5,i,f'{v:.1f}',va='center',fontsize=10,fontweight='700',color='#e8eaf6')
            ax_fc.set_xlabel('Mean Gain per Feature',fontsize=12,fontweight='700');ax_fc.set_title('Top 7 Feature Categories',fontsize=13,fontweight='700');pstyle_dark(ax_fc);plt.tight_layout();st.pyplot(fig_fc,use_container_width=True);plt.close();st.dataframe(fc_df.style.format({'Mean Gain':'{:.2f}','N Features Used':'{:,.0f}'}).background_gradient(subset=['Mean Gain'],cmap='Blues'),use_container_width=True,hide_index=True);st.markdown('<div class="info"><b>Chirality/charge</b> has highest mean gain despite only 36 features.</div>',unsafe_allow_html=True)

elif page=="Receptor Analysis":
    st.markdown("<h1>Receptor Analysis</h1>",unsafe_allow_html=True)
    CHEM={'NR-AR':{'f':['Steroidal scaffold (4 fused rings)','Aliphatic hydroxyl C17','R-stereocentre at C5'],'a':'C, O (hydroxyl)','p':'MW 250–350, logP 2–4, FracCSP3>0.6','act':'Testosterone, anabolic steroids, pesticides','note':'Very few toxic examples (309) → AUC 0.812'},'NR-AR-LBD':{'f':['Same as NR-AR','Higher LBD specificity'],'a':'C, O','p':'MW 250–400, logP 2–5','act':'Dihydrotestosterone analogues, flutamide','note':'72% co-activity with NR-AR → AUC 0.890'},'NR-AhR':{'f':['Flat polycyclic aromatic','Fused 6-6 rings','Halogenated aromatics'],'a':'C (aromatic), Cl, Br, F','p':'Low FracCSP3, 2+ aromatic rings, planar','act':'TCDD, PCBs, PAHs','note':'2D aromaticity fully captured by Morgan FP → AUC 0.913'},'NR-Aromatase':{'f':['Imidazole/triazole ring','Electron-withdrawing phenyl'],'a':'N (azole), C, halogens','p':'MW 300–500, logP 2–5, aromatic N','act':'Anastrozole, letrozole, fungicides','note':'Azole coordination to CYP19A1 heme → AUC 0.882'},'NR-ER':{'f':['Phenol group (A-ring)','Bisphenol motif','8–12Å inter-OH distance'],'a':'O (phenolic-OH), C (aromatic)','p':'MW 200–400, 1–3 aromatic rings, HBD≥1','act':'Estradiol, Bisphenol A, genistein, DES','note':'3D pharmacophore distance hard in 2D → AUC 0.748 (hardest)'},'NR-ER-LBD':{'f':['Same as NR-ER','LBD cavity fit'],'a':'O, C','p':'Compact, MW 200–350','act':'ERα-specific agonists','note':'75% co-activity with NR-ER → AUC 0.851'},'NR-PPAR-gamma':{'f':['Long aliphatic chain','Carboxylic acid head','Hydrophobic tail'],'a':'O (carboxylate), C','p':'MW 300–500, logP 3–6','act':'Thiazolidinediones, fibrates, plasticisers','note':'Most imbalanced (33.7:1) → AUC 0.888'},'SR-ARE':{'f':['Electrophilic warhead','Michael acceptor','Quinone motif'],'a':'O, C=C (electrophilic)','p':'Reactive groups, MW 150–400','act':'tBHQ, sulforaphane, quinones','note':'Nrf2 activation — thiol reactivity → AUC 0.857'},'SR-ATAD5':{'f':['DNA intercalation','Planar aromatic','Reactive electrophile'],'a':'N, C (planar)','p':'Planar, MW 200–500','act':'Doxorubicin analogues, alkylating agents','note':'58% co-activity with SR-p53 → AUC 0.918'},'SR-HSE':{'f':['Protein-denaturing','Heavy metal chelators'],'a':'S (thiol-reactive)','p':'Amphiphilic, reactive','act':'Cadmium, arsenic, proteotoxic compounds','note':'Heat shock — protein misfolding → AUC 0.842'},'SR-MMP':{'f':['Amphiphilic cation','Large hydrophobic surface'],'a':'N (positive), C (hydrophobic)','p':'logP>3, MW>300, amphiphilic','act':'Tricyclic antidepressants, mitochondrial uncouplers','note':'Physicochemical profile cleanly separates → AUC 0.934 (easiest)'},'SR-p53':{'f':['DNA strand break inducers','Alkylating agents'],'a':'Reactive electrophiles','p':'Reactive, MW 200–600','act':'Etoposide, camptothecin, nitroaromatics','note':'DNA damage → p53 stabilisation → AUC 0.904'}}
    sel=st.selectbox("Select receptor",TOX21_EP,key='rec');info=CHEM[sel];auc_v=float(ens_df[ens_df['endpoint']==sel]['ensemble_auc'].values[0])if not ens_df.empty else 0;m1,m2,m3,m4=st.columns(4);m1.metric("Ensemble AUC",f"{auc_v:.4f}");m2.metric("Threshold",f"{thresholds.get(sel,.5):.4f}");m3.metric("Toxic examples",f"{int(ens_df[ens_df['endpoint']==sel]['n_toxic'].values[0]):,}"if not ens_df.empty else"–");m4.metric("Imbalance",f"{spw.get(sel,0):.1f}:1")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="sh">Structural Features</div>',unsafe_allow_html=True)
        for feat in info['f']:st.markdown(f"• {feat}")
        st.markdown(f"**Key atoms:** {info['a']}");st.markdown(f"**Property profile:** {info['p']}");st.markdown(f"**Known activators:** {info['act']}")
    with c2:
        st.markdown('<div class="sh">Biology & Prediction Context</div>',unsafe_allow_html=True)
        st.markdown(f"**Mechanism:** {EP_BIO[sel][1]}");st.markdown(f"**Prediction note:** {info['note']}")
    st.markdown('<div class="sh">Threshold Heatmap</div>',unsafe_allow_html=True)
    if thresholds:fig_th,ax_th=plt.subplots(figsize=(13,2.3));fig_th.patch.set_facecolor('#1a1a2e');ax_th.set_facecolor('#1a1a2e');tv=np.array([[thresholds.get(ep,.5)for ep in TOX21_EP]]);im=ax_th.imshow(tv,cmap='RdYlGn',aspect='auto',vmin=0,vmax=.5);ax_th.set_xticks(range(len(TOX21_EP)));ax_th.set_xticklabels(TOX21_EP,rotation=35,ha='right',fontsize=10.5,fontweight='600',color='#e8eaf6');ax_th.set_yticks([])
    for j,ep in enumerate(TOX21_EP):v=thresholds.get(ep,.5);ax_th.text(j,0,f'{v:.3f}',ha='center',va='center',fontsize=9.5,fontweight='700',color='white'if v<.1 else'#1a1a2e')
    plt.colorbar(im,ax=ax_th,label='Threshold');ax_th.set_title('Decision threshold per endpoint',fontsize=12,fontweight='700',color='#4fc3f7');plt.tight_layout();st.pyplot(fig_th,use_container_width=True);plt.close()
    c1,c2=st.columns(2)
    for col,eps in[(c1,[e for e in TOX21_EP if e.startswith('NR-')]),(c2,[e for e in TOX21_EP if e.startswith('SR-')])]:
        with col:
            label='Nuclear Receptors'if col==c1 else'Stress Response Pathways';st.markdown(f'<div class="sh">{label}</div>',unsafe_allow_html=True)
            for ep in eps:
                av=float(ens_df[ens_df['endpoint']==ep]['ensemble_auc'].values[0])if not ens_df.empty else 0
                with st.expander(f"**{ep}** — {EP_BIO[ep][0]}  (AUC {av:.3f})"):st.markdown(EP_BIO[ep][1]);st.markdown(CHEM[ep]['note']);st.markdown(f"Threshold: `{thresholds.get(ep,.5):.4f}` · Imbalance: `{spw.get(ep,0):.1f}:1`")

elif page=="Multi-Assay Profile":
    st.markdown("<h1>Multi-Assay Profile & Co-Toxicity</h1>",unsafe_allow_html=True);ta,tb=st.tabs(["Co-Toxicity Heatmap","Compound Profiles"])
    with ta:
        st.markdown('<div class="sh">Co-Toxicity Correlation</div>',unsafe_allow_html=True);corr={'NR-AR':[1.00,0.72,0.18,0.35,0.41,0.68,0.29,0.21,0.15,0.12,0.19,0.17],'NR-AR-LBD':[0.72,1.00,0.14,0.28,0.35,0.61,0.24,0.18,0.12,0.10,0.16,0.14],'NR-AhR':[0.18,0.14,1.00,0.22,0.25,0.16,0.19,0.31,0.22,0.18,0.28,0.24],'NR-Aromatase':[0.35,0.28,0.22,1.00,0.48,0.39,0.31,0.19,0.13,0.11,0.22,0.18],'NR-ER':[0.41,0.35,0.25,0.48,1.00,0.75,0.38,0.22,0.17,0.14,0.25,0.21],'NR-ER-LBD':[0.68,0.61,0.16,0.39,0.75,1.00,0.32,0.20,0.14,0.11,0.21,0.18],'NR-PPAR-gamma':[0.29,0.24,0.19,0.31,0.38,0.32,1.00,0.24,0.18,0.15,0.27,0.22],'SR-ARE':[0.21,0.18,0.31,0.19,0.22,0.20,0.24,1.00,0.44,0.38,0.52,0.48],'SR-ATAD5':[0.15,0.12,0.22,0.13,0.17,0.14,0.18,0.44,1.00,0.35,0.41,0.58],'SR-HSE':[0.12,0.10,0.18,0.11,0.14,0.11,0.15,0.38,0.35,1.00,0.36,0.32],'SR-MMP':[0.19,0.16,0.28,0.22,0.25,0.21,0.27,0.52,0.41,0.36,1.00,0.45],'SR-p53':[0.17,0.14,0.24,0.18,0.21,0.18,0.22,0.48,0.58,0.32,0.45,1.00]};cdf=pd.DataFrame(corr,index=TOX21_EP)
        fig_h,ax_h=plt.subplots(figsize=(12,9));fig_h.patch.set_facecolor('#1a1a2e');sns.heatmap(cdf,annot=True,fmt='.2f',cmap='RdBu_r',center=0,vmin=-.1,vmax=1.,linewidths=.8,annot_kws={'size':10.5,'fontweight':'700','color':'#1a1a2e'},ax=ax_h,square=True,cbar_kws={'label':'Co-toxicity correlation'});ax_h.set_title('Co-toxicity Correlation',fontsize=14,fontweight='700',color='#4fc3f7');ax_h.tick_params(axis='both',labelsize=10.5,colors='#e8eaf6');plt.tight_layout();st.pyplot(fig_h,use_container_width=True);plt.close()
        st.markdown('<div class="info">🔗 <b>NR-AR/NR-AR-LBD</b> (r=0.72) &amp; <b>NR-ER/NR-ER-LBD</b> (r=0.75) share most activating compounds. <b>SR-ATAD5/SR-p53</b> (r=0.58) share genotoxic compounds. <b>NR-AhR</b> largely independent.</div>',unsafe_allow_html=True)
        if not ens_df.empty:st.markdown('<div class="sh">Stacked Distribution</div>',unsafe_allow_html=True);fig_s,ax_s=plt.subplots(figsize=(13,5));fig_s.patch.set_facecolor('#1a1a2e');ax_s.set_facecolor('#1a1a2e');x=np.arange(len(ens_df));w=.68;nm=ens_df['endpoint'].str.startswith('NR-').values;cs=['#c7d2fe'if m else'#bbf7d0'for m in nm];ct=['#667eea'if m else'#22c55e'for m in nm];tv=ens_df['n_toxic'].values;sv=ens_df['n_rows'].values-tv;ax_s.bar(x,sv,w,color=cs,alpha=.88,label='Safe',edgecolor='white',linewidth=1.5);ax_s.bar(x,tv,w,color=ct,alpha=.92,bottom=sv,label='Toxic',edgecolor='white',linewidth=1.5)
        for i,(t,s)in enumerate(zip(tv,sv)):pct=100*t/(t+s);ax_s.text(i,t+s+50,f'{pct:.1f}%',ha='center',fontsize=9,fontweight='700',color='#e8eaf6')
        ax_s.set_xticks(x);ax_s.set_xticklabels(ens_df['endpoint'],rotation=38,ha='right',fontsize=10.5,fontweight='600');ax_s.set_ylabel('Compounds',fontsize=12,fontweight='700');ax_s.legend(fontsize=11,framealpha=0.9,facecolor='#16213e',edgecolor='#667eea',labelcolor='#e8eaf6');ax_s.set_title('Compounds per endpoint',fontsize=13,fontweight='700');pstyle_dark(ax_s);plt.tight_layout();st.pyplot(fig_s,use_container_width=True);plt.close()
    with tb:
        KNOWN={"Bisphenol A":"CC(C)(c1ccc(O)cc1)c1ccc(O)cc1","TCDD":"Clc1cc2oc3cc(Cl)c(Cl)cc3oc2cc1Cl","Tamoxifen":"CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1","Testosterone":"CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C","Aspirin":"CC(=O)Oc1ccccc1C(=O)O","Rotenone":"COc1ccc2c(c1OC)CC(c1coc3cc(OC)c(OC)cc13)C(=O)O2"};sc=st.selectbox("Select",list(KNOWN.keys()));smi_c=KNOWN[sc];mc=Chem.MolFromSmiles(smi_c)
        if mc and models.get('xgb')and feat_cols:
            ci,cp=st.columns([1,2])
            with ci:st.image(mol_img(mc,(320,240)),caption=sc,use_container_width=True);mwc=Descriptors.MolWt(mc);lpc=Descriptors.MolLogP(mc);st.markdown(f"**MW:** {mwc:.1f} · **logP:** {lpc:.2f}")
            with cp:
                with st.spinner("Predicting…"):pc=predict(mc,models['xgb'],thresholds,feat_cols)
                prof=[{'Endpoint':ep,'Prob':r['prob'],'Label':r['label']}for ep in TOX21_EP if(r:=pc.get(ep))];pf=pd.DataFrame(prof);fig_pr,ax_pr=plt.subplots(figsize=(7,5));fig_pr.patch.set_facecolor('#1a1a2e');ax_pr.set_facecolor('#1a1a2e');bc_=['#ef4444'if l=='TOXIC'else'#667eea'for l in pf['Label']];ax_pr.barh(pf['Endpoint'],pf['Prob'],color=bc_,alpha=.92,edgecolor='white',linewidth=1.5)
                for i,(p,l)in enumerate(zip(pf['Prob'],pf['Label'])):ax_pr.text(p+.012,i,f'{p:.3f}',va='center',fontsize=9.5,fontweight='700',color='#e8eaf6')
                ax_pr.axvline(.5,color='#ffffff',ls='--',lw=2,alpha=.7,label='0.5 baseline');ax_pr.set_xlabel('Toxicity probability',fontsize=12,fontweight='700');ax_pr.set_title(f'Profile — {sc}',fontsize=12,fontweight='700');ax_pr.set_xlim(0,1.15);tp2=mpatches.Patch(color='#ef4444',label='Above threshold');sp2=mpatches.Patch(color='#667eea',label='Below threshold');ax_pr.legend(handles=[tp2,sp2],fontsize=10,framealpha=0.9,facecolor='#16213e',edgecolor='#667eea',labelcolor='#e8eaf6');pstyle_dark(ax_pr);plt.tight_layout();st.pyplot(fig_pr,use_container_width=True);plt.close()
                fl=[r['Endpoint']for _,r in pf.iterrows()if r['Label']=='TOXIC']
                if fl:st.markdown(f'<div class="err"><b>{len(fl)} endpoints above threshold:</b> {", ".join(fl)}</div>',unsafe_allow_html=True)
                else:st.markdown('<div class="ok">All endpoints below threshold.</div>',unsafe_allow_html=True)