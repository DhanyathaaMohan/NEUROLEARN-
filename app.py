
import streamlit as st
import pandas as pd, numpy as np, os, subprocess
from synthetic_data import generate_synthetic_interactions
from model_utils import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model as keras_load

st.set_page_config(page_title='NeuroLearn — Demo', layout='wide')

st.title('NeuroLearn — Personalized Learning (Demo)')

menu = st.sidebar.selectbox('View', ['Student View','Teacher Dashboard','Model & Data'])

if menu == 'Student View':
    st.header('Student View — Adaptive Content')
    st.write('Generate or upload synthetic data, then ask the model for a recommendation.')
    n_students = st.number_input('Number of students to simulate', min_value=1, max_value=200, value=20)
    n_sessions = st.number_input('Sessions per student', min_value=1, max_value=100, value=20)
    if st.button('Generate synthetic data'):
        df = generate_synthetic_interactions(n_students, n_sessions)
        st.session_state['df'] = df
        st.success('Synthetic data generated and stored in session state.')
    if 'df' in st.session_state:
        df = st.session_state['df']
        sid = st.selectbox('Pick student', sorted(df['student_id'].unique()))
        st.write(df[df['student_id']==sid].head())
        st.write('Simple recommendation: pick a content_id that historically had content_fit=1 for this student.')
        hist = df[df['student_id']==sid].groupby('content_id')['content_fit'].mean().sort_values(ascending=False)
        st.write(hist.head(5))
        st.write('Top recommendation: content_id =', int(hist.index[0]))

elif menu == 'Teacher Dashboard':
    st.header('Teacher Dashboard — Interventions & A/B Testing')
    st.write('This is a lightweight dashboard: shows per-student summary and allows quick A/B simulation.')
    if 'df' not in st.session_state:
        st.info('Generate synthetic data first in Student View or click below.')
        if st.button('Generate demo data (20 students)'):
            st.session_state['df'] = generate_synthetic_interactions(20, 20)
            st.success('Demo data stored.')
    if 'df' in st.session_state:
        df = st.session_state['df']
        agg = df.groupby('student_id').agg({'score':['mean','std'],'time_spent':'mean','content_fit':'mean'})
        agg.columns = ['mean_score','std_score','mean_time','mean_fit']
        st.dataframe(agg.reset_index())
        st.markdown('---')
        st.subheader('A/B test simulator')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Variant A: baseline content')
        with col2:
            st.write('Variant B: simplified content (simulated lift +0.08 for those with mean_fit<0.6)')
        if st.button('Run A/B test (simulate)'):
            import numpy as np
            a = df['score'].mean()
            # simulate lift for low-fit students
            df2 = df.copy()
            low = df2.groupby('student_id')['content_fit'].mean() < 0.6
            low_ids = low[low].index.tolist()
            mask = df2['student_id'].isin(low_ids)
            df2.loc[mask,'score'] = (df2.loc[mask,'score'] + 0.08).clip(0,1)
            b = df2['score'].mean()
            st.write('Avg score A:', round(a,3), 'Avg score B:', round(b,3))
            st.success('A/B simulation complete.')
elif menu == 'Model & Data':
    st.header('Model & Data — Train / Load model')
    st.write('You can train the simple model (may require TensorFlow).')
    if st.button('Run quick train (3 epochs)'):
        st.write('Training... this will call train.py in a subprocess.')
        try:
            subprocess.check_call([sys.executable, 'train.py', '--epochs', '3'])
            st.success('Training finished. Model saved to models/simple_model.h5')
        except Exception as e:
            st.error('Training failed or TensorFlow not installed in this environment. See README.\n' + str(e))
    if os.path.exists('models/simple_model.h5'):
        st.success('Model found: models/simple_model.h5')
        try:
            m = keras_load('models/simple_model.h5')
            st.write('Model summary:')
            st.text(m.summary())
        except Exception as e:
            st.write('Could not load model within Streamlit: ', e)
