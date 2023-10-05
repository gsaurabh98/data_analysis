# -*- coding: utf-8 -*-
"""
Created on Wed July 06 13:06:16 2022
@author: saurabh
"""
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from graphs import *

# https://github.com/Jcharis/Streamlit_DataScience_Apps/blob/master/EDA_app_with_Streamlit_Components/app.py
# https://www.youtube.com/watch?v=zWiliqjyPlQ&ab_channel=JCharisTech


st.set_page_config(
    page_title="Data Analysis",
    page_icon="icon.png",
    layout="wide",
    # initial_sidebar_state="expanded",
    menu_items={
        # 'Get Help': '',
        # 'Report a bug': " ",
        # 'About': ""
    }
)
# hiding footer
hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title('Data Analysis')
st.markdown(
    '<span style=" color:Grey"><i>Data analysis involves manipulating, transforming, and visualizing data in order to infer meaningful insights from the results.!</i></span>',
    unsafe_allow_html=True)
st.text('')
st.text('')

# st.sidebar.markdown('<b>Upload Dataset</b>', unsafe_allow_html=True)
# st.sidebar.info('Upload file')
# st.sidebar.markdown(
#     '<span style=" font-size:13px" >Dataset must contain numeric features!</span>',
#     unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(
    "Choose a dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if os.path.splitext(uploaded_file.name)[1].lower() in [".xlsx", ".xls"]:
        dataframe = pd.read_excel(uploaded_file)
    else:
        dataframe = pd.read_csv(uploaded_file)

    columns = dataframe.columns.tolist()
    # if not len(columns) > 0:
    #     st.sidebar.markdown(
    #         '<span style="color:red; font-size:18px" >No numeric features found, please upload different dataset.</span>',
    #         unsafe_allow_html=True)
    # else:
    # st.markdown('<b>Dataset sample</b>', unsafe_allow_html=True)

    st.info('Dataset sample')
    st.table(dataframe.head())

    schema = get_schema(dataframe)
    statistics_obj = {}
    statistics_obj['schema'] = pd.DataFrame(schema)
    statistics_obj['dataframe'] = dataframe
    descriptive_statistic = calculate_statistics(
        statistics_obj)
    stats = pd.merge(schema, descriptive_statistic, on='colName', how='left')
    stats.rename(
        columns={'colName': 'feature', 'dataType': 'type'},
        inplace=True)
    st.info('Stats')
    st.table(stats.drop('memType', axis=1).set_index('feature'))

    #  "Choose report type ",
    #     ('Pandas Profile', 'Sweetviz', 'Custom Report')
    # )
    # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',
    #  unsafe_allow_html=True)

    # if method == 'Custom Report':
    features = st.sidebar.multiselect(
        "Select feature",
        columns
    )

    # features = st.sidebar.multiselect('Select features', columns, key=None)
    if st.sidebar.button('Submit', key="method"):
        with st.spinner("Generating chart..."):
            if not features:
                st.sidebar.warning('Please select feature(s)')
            elif len(features) == 1:
                if schema.loc[schema['colName'].isin(features),
                              'dataType'].item() in ['Integer', 'Decimal']:
                    st.info('Histogram')
                    st.plotly_chart(get_histogram(dataframe, features[0]))
                    st.info('Boxplot')
                    st.plotly_chart(get_boxplot(dataframe, features[0]))
                elif schema.loc[schema['colName'].isin(features), 'dataType'].item() in ['Boolean', 'Categorical']:
                    st.info('Bar Chart')
                    st.plotly_chart(get_barplot(dataframe, features[0]))
                    st.info('Pie Chart')
                    st.plotly_chart(get_pieplot(dataframe, features[0]))
            elif len(features) > 1:
                if schema.loc[schema['colName'] == (features[0]), 'dataType'].item() in ['Integer', 'Decimal'] \
                        or schema.loc[schema['colName'] == (features[1]), 'dataType'].item() in ['Integer', 'Decimal']:
                    st.info('Histogram')
                    st.plotly_chart(
                        get_histogram(
                            dataframe, features[0],
                            features[1]))
                    st.info('Scatterplot')
                    st.plotly_chart(
                        get_scatterplot(
                            dataframe, features[0],
                            features[1]))
                elif schema.loc[schema['colName'] == (features[0]), 'dataType'].item() in ['Boolean', 'Categorical'] \
                        or schema.loc[schema['colName'] == (features[1]), 'dataType'].item() in ['Boolean', 'Categorical']:
                    st.info('Bar Chart')
                    st.plotly_chart(
                        get_barplot(
                            dataframe, features[0],
                            features[1]))
