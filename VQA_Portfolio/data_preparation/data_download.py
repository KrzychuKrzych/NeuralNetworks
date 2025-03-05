import os
import zipfile
import json
import requests
import pandas as pd

# Tworzenie struktury folderów
def create_directory_structure():
    """Tworzy strukturę folderów dla projektu VQA"""
    # Główny folder projektu
    os.makedirs("VQA_Portfolio", exist_ok=True)
    
    # Podfoldery
    folders = [
        "data_preparation",
        "model",
        "evaluation",
        "utils",
        "data/raw",
        "data/processed",
        "data/models",
        "data/results"
    ]
    
    # Tworzenie podfolderów w głównym folderze projektu
    for folder in folders:
        os.makedirs(os.path.join("VQA_Portfolio", folder), exist_ok=True)
    
    print("Struktura folderów została utworzona.")

# Tworzenie struktury folderów
create_directory_structure()

# URL do danych
vqa_v2_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
annotations_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" 
images_url = "http://images.cocodataset.org/zips/train2014.zip"

# Funkcja do pobierania plików
def download_file(url, filename, save_path):
    """Pobiera plik z podanego URL i zapisuje go pod wskazaną nazwą"""
    full_path = os.path.join(save_path, filename)
    
    # Sprawdź, czy plik już istnieje
    if os.path.exists(full_path):
        print(f"Plik {filename} już istnieje. Pomijam pobieranie.")
        return full_path
    
    print(f"Pobieranie {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(full_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                # Wyświetlanie postępu
                done = int(50 * downloaded / total_size)
                print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f}/{total_size/1024/1024:.2f} MB", end="")
    print(f"\nPobrano {filename}")
    return full_path

# Funkcja do rozpakowywania plików
def unzip_file(zip_path, extract_to):
    """Rozpakowuje plik ZIP do wskazanego folderu"""
    filename = os.path.basename(zip_path)
    print(f"Rozpakowywanie {filename}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Rozpakowano {filename}")

def main():
    # Ścieżka do folderu z danymi
    data_path = os.path.join("VQA_Portfolio", "data", "raw")
    
    # Pobieranie plików
    questions_zip = download_file(vqa_v2_url, "questions.zip", data_path)
    annotations_zip = download_file(annotations_url, "annotations.zip", data_path)
    
    # Pytanie użytkownika o pobieranie obrazów (mogą być duże)
    download_images = input("Czy chcesz pobrać obrazy treningowe? Plik ma około 14GB (tak/nie): ").lower()
    
    if download_images == 'tak':
        images_zip = download_file(images_url, "images.zip", data_path)
        # Rozpakowywanie obrazów
        unzip_file(images_zip, data_path)
    
    # Rozpakowywanie plików z pytaniami i adnotacjami
    unzip_file(questions_zip, data_path)
    unzip_file(annotations_zip, data_path)
    
    # Wyświetlanie zawartości folderów
    print("\nZawartość folderu z danymi:")
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")
    
    print("\nPobieranie i rozpakowywanie danych zakończone!")

if __name__ == "__main__":
    main()
