# -*- coding: utf-8 -*-
"""
Created on Thu aug 20 17:13:24 2022
Author: Saurabh Sharma
"""
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
import cufflinks as cf
cf.go_offline()

def get_histogram(dataframe, feature1, feature2=None):

    """
    Computes histogram on numeric features
        Parameters
        ------
        dataframe: pd.dataframe
            dataframe
        feature1: str
            feature name
        feature2: str
            feature name
        Returns
        ------
        fig: object
            plotly figure

    """

    fig = go.Figure()
    # fig = ff.create_distplot([dataframe[feature1].dropna()], [feature1],
    # show_hist=False, curve_type='kde')
    fig.add_trace(go.Histogram(
                            x=dataframe[feature1].dropna(),
                            name=feature1,
                        )
                        )
    if feature2:
        fig.add_trace(go.Histogram(
                            x=dataframe[feature2].dropna(),
                            name=feature2,
                        )
                        )
    fig.layout.update(
        dict(
            # width=500,
            # height=500,
            xaxis=dict(title=feature1,
                        showline=True,
                        showgrid=False),
            yaxis=dict(title='Values',
                        showgrid=False,
                        showline=True),
            showlegend=True,
        )
    )
    return fig

def get_boxplot(dataframe, feature1, feature2=None):

    """
    Computes boxplot on numeric features
        Parameters
        ------
        dataframe: pd.dataframe
            dataframe
        feature1: str
            feature name
        feature2: str
            feature name
        Returns
        ------
        fig: object
            plotly figure

    """

    fig = go.Figure()
    fig.add_trace(go.Box
                    (x=dataframe[feature1].dropna(),
                    name=feature1,
                    jitter=0.3,
                    pointpos=-1.8,
                    boxpoints='all',
                    boxmean='sd')
                    )
    fig.layout.update(
        dict(
            xaxis=dict(title='Values',
                        showline=True,
                        showgrid=False),
            yaxis=dict(
                showgrid=False,
                zeroline=True,
                showline=True),
            # paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
        )
    )
    # fig.update_traces(opacity=1)
    return fig


def get_barplot(dataframe, feature1, feature2=None):
    """
        Computes barplot on categorical features
            Parameters
            ------
            dataframe: pd.dataframe
                dataframe
            feature1: str
                feature name
            feature2: str
                feature name
            Returns
            ------
            fig: object
                plotly figure

        """
    fig = go.Figure()
    fig.add_trace(go.Bar(
    name =feature1,
    x = dataframe[feature1].value_counts().index,
    y = dataframe[feature1].value_counts()
    ))
    if feature2:
        fig.add_trace(go.Bar(
        name = feature2,
        x = dataframe[feature2].value_counts().index,
        y = dataframe[feature2].value_counts()
        ))
    layout = dict(
        showlegend=True,
        barmode='group',
        xaxis=dict(title=feature1,
                    showline=True,
                    showgrid=False),
        yaxis=dict(title='count',
                    showgrid=False,
                    zeroline=True,
                    showline=True)
                    )
    fig.layout.update(layout)
    return fig
    
def get_pieplot(dataframe, feature1):
    """
    Computes pieplot on categorical features
        Parameters
        ------
        dataframe: pd.dataframe
            dataframe
        feature1: str
            feature name
        Returns
        ------
        fig: object
            plotly figure

    """
    labels = dataframe[
                feature1].value_counts().index.tolist()
    values = dataframe[feature1].value_counts()
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=labels, values=values,
                            textinfo='value+percent'))
    fig.layout.update(
        dict(
            # autosize=False,
            # width=500,
            # height=500,
            # paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            xaxis=dict(
                showline=True,
                showgrid=False),
            yaxis=dict(
                showgrid=False,
                zeroline=True,
                showline=True),
        )
    )
    return fig

def get_scatterplot(dataframe, feature1, feature2):

    fig = go.Figure()
    fig.add_trace(go.Scatter
                    (x=dataframe[feature1].dropna(),
                    y=dataframe[feature2].dropna(),
                    mode='markers'
                    )
                    )

    fig.layout.update(
        dict(
            # title='{0} VS {1}' ''.format(feature1,
            #                                 feature2),
            xaxis=dict(title=feature1,
                        showgrid=False),
            yaxis=dict(title=feature2,
                        showgrid=False,
                        zeroline=True),
            # paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )
    )
    return fig

def get_schema(dataframe):

    """
    Computes schema for all the features
        Parameters
        ------
        dataframe: pd.dataframe
            dataframe
        Returns
        ------
        dtypes: pd.DataFrame
            datatypes dataframe

    """

    dtypes = dataframe.dtypes.map(str)
    dtypes = dtypes.reset_index()
    dtypes.columns = ['colName', 'memType']
    numerics = dtypes[(dtypes.memType.str.find("int") > -1) | \
                        (dtypes.memType.str.find(
                            "float") > -1)].colName.to_list()
    objects = dtypes[dtypes.memType == 'object'].colName.to_list()
    dates = dtypes[
        dtypes.memType.str.find("datetime") > -1].colName.to_list()
    bools = dtypes[dtypes.memType == 'bool'].colName.to_list()
    dtypes['dataType'] = None
    dtypes.loc[dtypes.colName.isin(dates), "dataType"] = "Datetime"
    dtypes.loc[dtypes.colName.isin(bools), "dataType"] = "Boolean"

    # Check for boolean
    for numeric_col in numerics:
        if set(dataframe[numeric_col].unique()) == {0, 1}:
            if dataframe[numeric_col].isnull().sum() == 0:
                dataframe[numeric_col] = dataframe[
                    numeric_col].astype("uint8")
            dtypes.loc[
                dtypes.colName == numeric_col, "dataType"] = "Boolean"
        else:
            dtype = str(dataframe[numeric_col].dtype)
            dtypes.loc[dtypes.colName == numeric_col, "dataType"] = \
                "Integer" if dtype.find("int") > -1 else "Decimal"

    # Check for date, numeric and text type
    for obj_col in objects:
        try:
            if dataframe[obj_col].dropna().map(
                    lambda x: isinstance(x, bool)).value_counts()[True]:
                dataframe.loc[dataframe[obj_col].isnull() == False,
                                obj_col] = dataframe.loc[
                    dataframe[obj_col].isnull() == False, obj_col] \
                    .astype(bool)
                dtypes.loc[
                    dtypes.colName == obj_col, "dataType"] = "Boolean"
                continue
        except:
            pass
        try:
            dataframe[obj_col] = pd.to_datetime(dataframe[obj_col],
                                                utc=True)
            dataframe[obj_col] = dataframe[obj_col].apply(
                lambda x: x.replace(tzinfo=None))
            dtypes.loc[
                dtypes.colName == obj_col, "dataType"] = "Datetime"
        except:
            try:
                temp_df = dataframe[obj_col].replace(
                    '[\,]', '', regex=True).astype(float)
                dtypes.loc[
                    dtypes.colName == obj_col, "dataType"] = "Decimal"
                dataframe[obj_col] = temp_df
            except:
                if all(dataframe[obj_col].str.startswith(
                        tuple(['http:', 'https:']))):
                    dtypes.loc[
                        dtypes.colName == obj_col, "dataType"] = "URL"
                else:
                    cp_mean = dataframe[obj_col].dropna().map(
                        lambda x: len(str(x))).mean()
                    cp_words = dataframe[obj_col].dropna().map(
                        lambda x: len(str(x).split(" "))).mean()

                    if cp_mean >= 100 or \
                            cp_words >= 10:
                        dtypes.loc[
                            dtypes.colName == obj_col, "dataType"] = "Text"
                    else:
                        dtypes.loc[dtypes.colName == obj_col,
                                    "dataType"] = "Categorical"

    # dtypes.drop('memType',axis=1, inplace=True)
    # dtypes.columns = ['feature', 'type']
    return dtypes


def calculate_statistics(stats_obj):
    """
        Computes descriptive stats of a dataframe
        ------
        Parameters:
            stats_obj: dictionary
                schema: dataframe
                    schema of a dataframe
                datafram: pandas dataframe
                    dataframe on which descriptive stats will be computed
        Returns:
        ------
            descriptive stats: dictionary
                dictionary of feature wise descriptive stats
        Exceptions
        ------
            Raises exception if fails to computes stats
    """

    schema = stats_obj['schema']
    dataframe = stats_obj['dataframe']
    emp_df = pd.DataFrame()

    try:

        # Compute date statistics and append to sk.pipeline
        datetime_columns = schema.loc[
            schema['dataType'] == 'Datetime', "colName"].tolist()
        date_descriptive_statistics = pd.DataFrame()
        if datetime_columns:
            date_stats = dataframe[datetime_columns].describe()
            date_descriptive_statistics = date_stats.rename(
                index={
                    'first': 'min',
                    'last': 'max',
                })

            date_descriptive_statistics.loc['50%'] \
                = dataframe[datetime_columns].apply(lambda x: x.quantile(.5))

            try:
                date_descriptive_statistics.loc['frequency'] = dataframe[
                    datetime_columns].apply(compute_frequency)
            except Exception as e:
                # logger.info(e)
                date_descriptive_statistics.loc['frequency'] = None

            date_descriptive_statistics.loc['mean'] = \
                date_descriptive_statistics.loc['std'] = \
                date_descriptive_statistics.loc['25%'] = \
                date_descriptive_statistics.loc['75%'] = None

            date_descriptive_statistics = date_descriptive_statistics[
                ~date_descriptive_statistics.index.isin(
                    ['top', 'freq', 'unique'])]

        # Compute missing values
        df_info = dataframe.isnull().sum().to_frame("missing")

        # Compute descriptive statistics
        try:
            numeric_descriptive_statistics = dataframe.describe(
                include=np.number).round(decimals=5)
        except ValueError:
            # logger.exception("EDA - No numeric data")
            numeric_descriptive_statistics = pd.DataFrame(
                columns=list(dataframe.columns),
                index=[
                    "min", "max", "25%", "50%", "75%", "mean",
                    "count", "std"]
            )

        statistics = pd.merge(
            date_descriptive_statistics,
            numeric_descriptive_statistics,
            left_index=True, right_index=True, how="outer").T

        other = dataframe[
            dataframe.select_dtypes(
                exclude=['number', 'object']).columns].apply(
            lambda x: x.drop_duplicates().size)

        number = dataframe[
            dataframe.select_dtypes(include='number').columns].apply(
            lambda x: x.drop_duplicates().size)

        obj = dataframe[
            dataframe.select_dtypes(include='object').columns].apply(
            lambda x: x.dropna().drop_duplicates().size)

        uniques = pd.concat([number, obj, other])
        del other, number, obj

        statistics = pd.merge(statistics, uniques.to_frame("unique"),
                              left_index=True,
                              right_index=True, how='outer')
        statistics = pd.merge(df_info, statistics, left_index=True,
                              right_index=True, how='outer')

        statistics.drop(["count", "25%", "75%"], axis=1, inplace=True)

        statistics = statistics.reset_index().round(decimals=5)

        statistics.rename(columns={
            'index': 'colName',
            '50%': 'median'
        }, inplace=True)

        return statistics
    except Exception as e:
        # logger.exception(f'error at computing descriptive stats {e}')
        print('e',e)
        return emp_df


def compute_frequency(date_column_series):
    """
        Computes frequency of timeseries dataset
        ------
        Parameters:
            date_column_series : pandas series
                series with date specified column
        Returns:
        ------
            frequency: str
                frequency of a dataset
        Exceptions
        ------
            No exception
    """
    freq_series = date_column_series.sort_values().dt.date
    difference = freq_series - freq_series.shift(1)
    days = difference.mode().iloc[0].days

    if days > 92:
        frequency = 'YS (Year start)'
    elif 92 <= days > 31:
        frequency = 'QS (Quarter start)'
    elif 31 <= days > 1:
        frequency = 'MS (Month start)'
    elif 7 <= days > 1:
        frequency = 'W (Weekly)'
    elif days == 1:
        frequency = 'D (Calendar day)'
    else:
        frequency = None

    return frequency