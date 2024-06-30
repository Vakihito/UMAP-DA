from umap.umap_ import UMAP
import numpy as np

n_feat = 3
n_data = 6
x = np.random.rand(n_data, n_feat)
y = [np.random.randint(0, 2) for _ in range(n_data)]
domains = [np.random.randint(0, 2) for _ in range(n_data)]

print("starting umap")
umap_model = UMAP(n_neighbors=2, n_components=2,random_state=0)
print("x : ", x)
print("y : ", y)
print("domains : ", domains)

return_data = umap_model.fit_transform(x, y=y,domains=domains)
print("umap return")
print(return_data)