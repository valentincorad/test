# https://dashboard-marie.herokuapp.com

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import requests
import pickle
from sklearn.neighbors import KDTree
import lightgbm


def main():

    # LOAD DATA
    # Reduce size of data
	n_rows=1000
	@st.cache(allow_output_mutation=True)
	def transform_raw_data(path):
		raw_data = pd.read_csv(path, nrows=n_rows, index_col='SK_ID_CURR')

		raw_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
		raw_data = raw_data[raw_data['CODE_GENDER'] != 'XNA']
		raw_data = raw_data[raw_data['AMT_INCOME_TOTAL'] < 100000000]

		good_cols = ['CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
		             'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE']
		infos = raw_data.loc[:,good_cols]

		infos['AGE'] = (infos['DAYS_BIRTH']/-365).astype(int)
		infos['YEARS EMPLOYED'] = round((infos['DAYS_EMPLOYED']/-365), 2)
		infos.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

		infos = infos[[ 'AGE', 'CODE_GENDER','NAME_FAMILY_STATUS',
		               'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE','YEARS EMPLOYED',
		               'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
		             ]]

		infos.columns = [ 'AGE', 'GENDER','FAMILY STATUS',
		               'EDUCATION TYPE', 'OCCUPATION TYPE','YEARS EMPLOYED',
		               'YEARLY INCOME', 'AMOUNT CREDIT', 'AMOUNT ANNUITY', 'GOODS PRICE',
		             ]
		return infos

	@st.cache(allow_output_mutation=True)
	def load_app_train_clean():
		app_train_clean = pd.read_csv('./data_model/app_train_clean1.csv', nrows=n_rows, index_col=0)
		return app_train_clean.drop('TARGET', axis=1)


	data_load_state = st.text('Loading data...')
	infos = transform_raw_data('./data_model/application_train1.csv')
	data_processed = load_app_train_clean()
	moyennes = pd.read_csv('./data_model/moyennes.csv', index_col=0)
	with open('./data_model/light_gbm.pickle', 'rb') as file :
		LGB = pickle.load(file)
	df_vois = pd.get_dummies(infos.iloc[:,:6])
	df_vois = df_vois.dropna()
	tree = KDTree(df_vois)
	data_load_state.text('')

    # _____________________________________________________
    # GENERAL INFORMATION
	st.title('Dashboard PRET A DEPENSER')

    # Select client
	client_id = st.sidebar.selectbox('Select ID Client :', infos.index)

    # Display general informations in sidebar
	st.sidebar.table(infos.loc[client_id][:6])

    # Plot data relative to income and credit amounts
	bar_cols = infos.columns[6:10]
	infos.at['Moyenne clients', bar_cols] = infos.loc[:,bar_cols].mean()

	fig = go.Figure(data=[
	    go.Bar(name='Client sélectionné', x=bar_cols, y=infos.loc[client_id, bar_cols].values),
	    go.Bar(name='Moyenne des clients', x=bar_cols, y=infos.loc['Moyenne clients', bar_cols].values)
	])
	fig.update_layout(title_text=f'Montants des revenus et du crédit demandé pour le client {client_id}')

	st.plotly_chart(fig, use_container_width=True)

    # ________________________________________________________
    # PREDICTIONS

	st.header('Risque de défaut')
	# data client
	data_client = data_processed.loc[client_id:client_id]
	prediction_client = 100*LGB.predict_proba(data_client)[0][1]

    # Get predictions for similar clients :
    # get indexes of 10 nearest neighbors
	idx_vois = tree.query([df_vois.loc[client_id].fillna(0)], k=10)[1][0]
	# select processed data of neighbors
	data_vois = data_processed.iloc[idx_vois]
	#make predictions
	prediction_voisins = 100*LGB.predict_proba(data_vois).mean(axis=0)[1]


    # Plot gauge
	gauge = go.Figure(go.Indicator(
        mode = "gauge+delta+number",
        value = prediction_client,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 25], 'color': "lightgreen"},
                     {'range': [25, 50], 'color': "lightyellow"},
                     {'range': [50, 75], 'color': "orange"},
                     {'range': [75, 100], 'color': "red"},
                     ],
                 'threshold': {
                'line': {'color': "black", 'width': 10},
                'thickness': 0.8,
                'value': prediction_client},

                 'bar': {'color': "black", 'thickness' : 0.2},
                },
        delta = {'reference': prediction_voisins,
        'increasing': {'color': 'red'},
        'decreasing' : {'color' : 'green'}}
        ))

	st.plotly_chart(gauge)

	st.markdown('Pour le client sélectionné : **{0:.1f}%**'.format(prediction_client))
	st.markdown('Pour les clients similaires : **{0:.1f}%** (critères de similarité : âge, genre,\
	     statut familial, éducation, profession, années d\'ancienneté)'.format(prediction_voisins))


    # ________________________________________________________
    # INTERPRETATION

	feature_desc = { 'EXT_SOURCE_2' : 'Score normalisé attribué par un organisme indépendant',
	                'EXT_SOURCE_3' :  'Score normalisé attribué par un organisme indépendant',
	                'AMT_ANNUITY' : 'Montant des annuités',
	                'AMT_GOODS_PRICE' : 'Montant du bien immobilier',
	                'CREDIT_INCOME_PERCENT' : 'Crédit demandé par rapport aux revenus',
	                'DAYS_EMPLOYED_PERCENT' : 'Années travaillées en pourcentage' }

	st.header('Interprétation du résultat')
	feature = st.selectbox('Selectionnez la variable à comparer', moyennes.columns)

	# get mean of features for neighbors
	mean_vois = pd.DataFrame(data_vois.mean(), columns=['voisins']).T

	# Compare features
	dfcomp = pd.concat([moyennes, mean_vois, data_client], join = 'inner').round(2)

	fig2 = go.Figure(data=[go.Bar(
	    x=dfcomp[feature],
	    y=['Moyenne des clients en règle ',
		  'Moyenne des clients en défaut ',
		  'Moyenne des clients similaires ',
		  'Client Sélectionné '],
	    marker_color=['green','red', 'orange', 'blue'],
	    orientation ='h'
	)])
	fig2.update_layout(title_text=feature_desc[feature])

	st.plotly_chart(fig2)

if __name__== '__main__':
	#
	# modif color branch
    main()
