# solvro-rekrutacja
## Cały kod jest przedstawiony w notatniku: model-prezentacja.ipynb
# obsługa folderów:
 Functions to zbiór funkcji używanych na przestrzeni projektu, nic z nim nie robimy

 hyperparameters jak sama nazwa wskazuje posiada hiperparametry do modelu

 data_loader transformuje obrazy i wczytuje je w batche
 
 model.py tworzy model i trenuje model, po czym zapisuje go do 
 pliku best_model.pth

 classify służy do klasyfikacji nowych, swoich zdjęć. Należy w zmiennej path ustawić swoją ścieżkę ze zdjęciem i odpalić kod