
import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import kagglehub
import os

print(os.listdir())
df = pd.read_csv("/content/features_30_sec.csv")
print(df.head())


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("/content/features_30_sec.csv")

# Drop unnecessary columns and prepare data
features = df.drop(columns=['filename', 'label'])

# Sidebar UI
st.sidebar.header("Music Genre Clustering")
num_clusters = st.sidebar.slider("Select number of clusters (K)", min_value=2, max_value=10, value=5)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features)

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Scatter plot of clusters
st.subheader("K-Means Clustering Visualization")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Cluster'], palette='tab10')
st.pyplot(fig)

# Show cluster results
st.subheader("Clustered Data")
st.write(df[['filename', 'label', 'Cluster']].head(20))

st.write("### Insights")
st.write("- Different genres are grouped into clusters.")
st.write("- Adjusting K changes the cluster formation.")

st.sidebar.write("Adjust K value for different clusters.")

!streamlit run app.py & npx localtunnel --port 8501

