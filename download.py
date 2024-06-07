import kaggle
kaggle.api.authenticate()

kaggle.api.dataset_download_files('stoicstatic/face-recognition-dataset', path = '.', unzip = True)