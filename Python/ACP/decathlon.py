import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# ---------------------
#        SIDEBAR
# ---------------------
st.sidebar.title("⚙️ Paramètres ACP")

uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, decimal=',', sep=";", engine='python')
else:
    st.write("Importez un CSV dans la sidebar.")
    st.stop()

# Choix des variables numériques
num_cols = df.select_dtypes(include=np.number).columns.tolist()
st.write("Types détectés :")
st.write(df.dtypes)
variables = st.sidebar.multiselect("Variables à inclure", num_cols, default=num_cols)

if len(variables) < 2:
    st.warning("Sélectionnez au moins 2 variables numériques.")
    st.stop()

# Nombre de composantes
n_components = st.sidebar.slider(
    "Nombre de composantes ACP",
    min_value=2,
    max_value=min(10, len(variables)),
    value=2
)

# ---------------------
#    PREPROCESSING
# ---------------------
X = df[variables].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=n_components)
coords = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_ * 100
loadings = pca.components_.T

# ---------------------
#      DASHBOARD
# ---------------------
st.title("📊 ACP Interactive")
st.markdown("Analyse complète : individus, variables, contributions, inertie…")

# ---------------------
#   1. INERTIA PLOT
# ---------------------
fig_inertia = px.bar(
    x=[f"Dim {i+1}" for i in range(n_components)],
    y=explained_var,
    labels={"x": "Composantes", "y": "% Variance expliquée"},
    title="Variance expliquée par composante"
)
st.plotly_chart(fig_inertia)

# ---------------------
#   2. INDIVIDUALS MAP
# ---------------------
st.subheader("🧍 Nuage des individus")

fig_ind = px.scatter(
    x=coords[:, 0],
    y=coords[:, 1],
    hover_name=X.index,
    labels={"x": f"Dim 1 ({explained_var[0]:.1f}%)",
            "y": f"Dim 2 ({explained_var[1]:.1f}%)"},
    title="Projection des individus"
)

# Ajout des lignes grises en épaisseur 2
fig_ind.add_shape(
    type="line",
    x0=0, x1=0,
    y0=min(coords[:, 1]), y1=max(coords[:, 1]),
    line=dict(color="gray", width=2)
)

fig_ind.add_shape(
    type="line",
    x0=min(coords[:, 0]), x1=max(coords[:, 0]),
    y0=0, y1=0,
    line=dict(color="gray", width=2)
)

# S'assurer que les axes ne coupent pas les lignes
fig_ind.update_layout(
    xaxis=dict(zeroline=False),
    yaxis=dict(zeroline=False)
)

st.plotly_chart(fig_ind)

# ---------------------
#   3. CORRELATION CIRCLE
# ---------------------
st.subheader("🎯 Cercle des corrélations (variables)")

theta = np.linspace(0, 2*np.pi, 200)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

fig_corr = go.Figure()

fig_corr.add_trace(go.Scatter(
    x=circle_x,
    y=circle_y,
    mode="lines",
    line=dict(color="lightgray"),
    showlegend=False
))

for i, var in enumerate(variables):
    fig_corr.add_trace(go.Scatter(
        x=[0, loadings[i, 0]],
        y=[0, loadings[i, 1]],
        mode="lines+markers+text",
        text=[None, var],
        textposition="top center"
    ))

fig_corr.update_layout(
    xaxis=dict(scaleanchor="y", range=[-1.1, 1.1]),
    yaxis=dict(range=[-1.1, 1.1]),
    title="Cercle des corrélations",
    width=700,
    height=700
)

st.plotly_chart(fig_corr)

# ---------------------
#   4. CONTRIBUTIONS
# ---------------------
st.subheader("📌 Contributions des variables")

contrib = (loadings[:, :2] ** 2).sum(axis=1)
contrib = contrib / contrib.sum() * 100

fig_contrib = px.bar(
    x=variables,
    y=contrib,
    title="Contribution des variables aux 2 premières composantes",
    labels={"x": "Variable", "y": "Contribution (%)"}
)
st.plotly_chart(fig_contrib)

# ---------------------
#   5. RAW OUTPUT
# ---------------------
with st.expander("🔍 Données PCA brutes"):
    st.dataframe(pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)]))
