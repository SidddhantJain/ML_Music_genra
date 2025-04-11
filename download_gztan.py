import kagglehub

# Download dataset from KaggleHub
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

print("✅ Dataset downloaded at:", path)
