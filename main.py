import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances

def dbscan(X, eps, min_samples):
    n = X.shape[0]
    visited = np.zeros(n, dtype=bool)
    clusters = []
    
    def expand_cluster(point_idx, neighbors):
        cluster = [point_idx]
        visited[point_idx] = True
        
        while neighbors:
            neighbor_idx = neighbors.pop()
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors.extend(new_neighbors)
            
            if neighbor_idx not in cluster:
                cluster.append(neighbor_idx)
        
        return cluster
    
    def region_query(point_idx):
        return [idx for idx, dist in enumerate(distances[point_idx]) if dist <= eps]
    
    distances = euclidean_distances(X)
    
    with open('dbscan_output.txt', 'w') as file:
        for iteration in range(n):
            if visited[iteration]:
                continue
            
            neighbors = region_query(iteration)
            
            if len(neighbors) < min_samples:
                file.write(f"Punto {iteration + 1}: Ruido\n")
                continue
            
            cluster = expand_cluster(iteration, neighbors)
            clusters.append(cluster)
            
            file.write(f"Iteracion: {len(clusters)}\n")
            file.write("Punto \t|\t X \t,\t Y \t|\t Visitado \t|\t Distancia a los demas puntos\n")
            file.write("---------------------------------------------------------------------------------\n")
            for i, (x, y) in enumerate(X):
                visited_mark = "X" if visited[i] else ""
                dist_to_all_points = " ".join([f"{distances[i][j]:.4f}" for j in range(len(X))])
                file.write(f"{i+1:5} \t|\t {x:.2f} \t,\t {y:.2f} \t|\t {visited_mark} \t|\t {dist_to_all_points}\n")
            file.write("---------------------------------------------------------------------------------\n")
    
    return clusters, visited

# Generar datos de muestra
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Parámetros de DBSCAN
eps = 0.75
min_samples = 5

# Ejecutar el algoritmo DBSCAN
clusters, visited = dbscan(X, eps, min_samples)

# Mostrar gráfico con los puntos y grupos
plt.scatter(X[:, 0], X[:, 1], c='blue')
for j, cluster in enumerate(clusters):
    cluster_points = X[cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Grupo {j+1}")
plt.legend()
plt.show()