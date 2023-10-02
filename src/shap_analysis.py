import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Subtract
import shap

from cognitive_prior_network import CognitivePriorNetwork

project_path = os.path.dirname(os.path.abspath(__file__))
base_features = ["Ha", "pHa", "La", "Hb", "pHb", "Lb", "LotNumB",
                 "LotShapeB", "Corr", "Amb", "Block", "Feedback"]

def combine_two_models(m1, m2):
    m_input = Input([12])
    choices_pred = m1(m_input)
    cpc_pred = m2(m_input)
    m_out = Subtract()([choices_pred, cpc_pred])
    m = Model(inputs=[m_input], outputs=[m_out])
    return m

def calc_diff_nn_shap_values(cpn1, cpn2):
    cpn1.model._name = "cpc_cpn"
    cpn2.model._name = "choices_cpn"
    diff_nn_model = combine_two_models(cpn1.model, cpn2.model)
    choices_df = pd.read_csv(project_path + "/../data/choices13k.csv", index_col=0)
    synth15 = pd.read_csv(project_path + "/../data/synth15.csv")
    synth15 = synth15.drop(['Rate'], axis=1).to_numpy().astype('float32')
    explainer = shap.GradientExplainer(diff_nn_model, synth15)
    diff_nn_shap_values = explainer.shap_values(choices_df[base_features].to_numpy(), nsamples=1000)
    diff_nn_shap_values = diff_nn_shap_values[0]
    return diff_nn_shap_values

if __name__ == "__main__":
    cpc_bourgin = CognitivePriorNetwork()
    cpc_bourgin.load(project_path + "/../models/cpc_bourgin_prior")
    choices_bourgin = CognitivePriorNetwork()
    choices_bourgin.load(project_path + "/../models/choices_bourgin")
    diff_nn_shap_values = calc_diff_nn_shap_values(cpc_bourgin, choices_bourgin)
    np.save(project_path + "/../data/diff_nn_shap_values_bourgin_choices_noprior.npy", diff_nn_shap_values)

