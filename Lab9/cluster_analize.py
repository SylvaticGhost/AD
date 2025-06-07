import pandas as pd
import numpy as np

def get_dispersion(df, clusters):
      dispersion_dict = {}
      centers_dict = {}

      for cluster_id in np.unique(clusters):
          cluster_data = df[clusters == cluster_id]

          centroid = cluster_data.mean()

          squared_distances = ((cluster_data - centroid) ** 2).sum(axis=1)

          dispersion = squared_distances.mean()

          dispersion_dict[cluster_id] = dispersion
          centers_dict[cluster_id] = centroid['All']

      dispersion_df = pd.DataFrame({
          'Cluster': list(dispersion_dict.keys()),
          'Size': [sum(clusters == c_id) for c_id in dispersion_dict.keys()],
          'Dispersion': list(dispersion_dict.values()),
          'Center': list(centers_dict.values()),
      })

      return dispersion_df