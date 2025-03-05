import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import sys

# Dodanie ścieżki głównego katalogu projektu do sys.path
sys.path.append("VQA_Portfolio")

# Importowanie własnego modułu
from model.vqa_model import VQAModel, VQADataset, get_transforms

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trenuje model przez jedną epokę.
    
    Args:
        model: Model do trenowania
        dataloader: DataLoader z danymi treningowymi
        criterion: Funkcja straty
        optimizer: Optymalizator
        device: Urządzenie (CPU/GPU)
    
    Returns:
        tuple: (średnia strata, dokładność)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Pasek postępu
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Przeniesienie danych na urządzenie
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        if 'answer_idx' in batch:
            labels = batch['answer_idx'].to(device)
            
            # Zerowanie gradientów
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statystyki
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Wyświetlanie paska postępu
        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(f"\rEpoka: [{batch_idx+1}/{total_batches}] Loss: {running_loss/(batch_idx+1):.4f} "
                  f"Acc: {100.*correct/total:.2f}%", end='')
    
    print()  # Nowa linia po pasku postępu
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Waliduje model na zbiorze walidacyjnym.
    
    Args:
        model: Model do walidacji
        dataloader: DataLoader z danymi walidacyjnymi
        criterion: Funkcja straty
        device: Urządzenie (CPU/GPU)
    
    Returns:
        tuple: (średnia strata, dokładność)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Przeniesienie danych na urządzenie
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if 'answer_idx' in batch:
                labels = batch['answer_idx'].to(device)
                
                # Forward pass
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Statystyki
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total if total > 0 else 0
    
    return val_loss, val_acc

def save_training_history(history, output_path):
    """
    Zapisuje historię treningu do pliku i tworzy wykresy.
    
    Args:
        history: Słownik z historią treningu
        output_path: Ścieżka folderu wyjściowego
    """
    # Tworzenie folderu, jeśli nie istnieje
    os.makedirs(output_path, exist_ok=True)
    
    # Zapisywanie danych historii do pliku CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_path, 'training_history.csv'), index=False)
    
    # Tworzenie wykresów
    plt.figure(figsize=(12, 5))
    
    # Wykres straty
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Wykres dokładności
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'training_history.png'))
    plt.close()
    
    print(f"Zapisano historię treningu do {output_path}")

def train_model(train_loader, val_loader, num_classes, device, epochs=5, 
                learning_rate=0.0001, output_dir="VQA_Portfolio/data/models"):
    """
    Trenuje model VQA.
    
    Args:
        train_loader: DataLoader z danymi treningowymi
        val_loader: DataLoader z danymi walidacyjnymi
        num_classes: Liczba klas (możliwych odpowiedzi)
        device: Urządzenie (CPU/GPU)
        epochs: Liczba epok
        learning_rate: Współczynnik uczenia
        output_dir: Katalog wyjściowy do zapisania modelu
    
    Returns:
        model: Wytrenowany model
        history: Historia treningu
    """
    # Inicjalizacja modelu
    model = VQAModel(num_classes=num_classes).to(device)
    
    # Funkcja straty i optymalizator
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Śledzenie historii treningu
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    # Utworzenie folderu na modele
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Rozpoczęcie treningu na urządzeniu: {device}")
    print(f"Liczba epok: {epochs}")
    print(f"Rozmiar batcha: {train_loader.batch_size}")
    print(f"Liczba klas: {num_classes}")
    print(f"Learning rate: {learning_rate}")
    
    # Trenowanie modelu
    best_val_acc = 0.0
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"\nEpoka {epoch+1}/{epochs}:")
        
        # Trenowanie
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Walidacja
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Czas trwania epoki
        epoch_time = time.time() - start_time
        history['epoch_time'].append(epoch_time)
        
        print(f"Epoka {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"Czas: {epoch_time:.2f}s")
        
        # Zapisywanie najlepszego modelu
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'vqa_model_best.pth'))
            print(f"Zapisano najlepszy model (val_acc: {val_acc:.2f}%)")
    
    # Zapisywanie ostatniego modelu
    torch.save(model.state_dict(), os.path.join(output_dir, 'vqa_model_last.pth'))
    
    # Zapisywanie konfiguracji treningu
    config = {
        'num_classes': num_classes,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': train_loader.batch_size,
        'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'best_val_acc': best_val_acc
    }
    
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Zapisywanie historii treningu
    save_training_history(history, output_dir)
    
    return model, history

def main():
    """Główna funkcja do trenowania modelu VQA"""
    
    # Ścieżki do folderów
    base_dir = "VQA_Portfolio"
    processed_dir = os.path.join(base_dir, "data", "processed")
    images_dir = os.path.join(base_dir, "data", "raw", "train2014")
    models_dir = os.path.join(base_dir, "data", "models")
    
    # Sprawdzenie, czy potrzebne pliki istnieją
    csv_path = os.path.join(processed_dir, "vqa_processed.csv")
    answer_map_path = os.path.join(processed_dir, "answer_to_idx.json")
    
    if not os.path.exists(csv_path):
        print(f"Błąd: Brak pliku z danymi {csv_path}. Najpierw uruchom data_preprocessing.py.")
        return
    
    if not os.path.exists(answer_map_path):
        print(f"Błąd: Brak pliku z mapowaniem odpowiedzi {answer_map_path}. Najpierw uruchom data_preprocessing.py.")
        return
    
    if not os.path.exists(images_dir):
        print(f"Błąd: Folder z obrazami {images_dir} nie istnieje.")
        return
    
    # Sprawdź liczbę klas
    with open(answer_map_path, 'r') as f:
        answer_to_idx = json.load(f)
    num_classes = len(answer_to_idx) + 1  # +1 dla nieznanych odpowiedzi
    
    # Transformacje dla obrazów
    train_transform, val_transform = get_transforms()
    
    # Wczytanie danych
    df = pd.read_csv(csv_path)
    print(f"Wczytano {len(df)} rekordów z {csv_path}")
    
    # Podział na zbiór treningowy i walidacyjny (80% / 20%)
    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    print(f"Podział danych: {train_size} próbek treningowych, {val_size} próbek walidacyjnych")
    
    # Zapisanie zbiorów do plików CSV
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    
    # Tworzenie datasetów
    train_dataset = VQADataset(
        os.path.join(processed_dir, "train.csv"),
        images_dir,
        transform=train_transform,
        answer_map_path=answer_map_path
    )
    
    val_dataset = VQADataset(
        os.path.join(processed_dir, "val.csv"),
        images_dir,
        transform=val_transform,
        answer_map_path=answer_map_path
    )
    
    # Parametry treningu
    batch_size = 16  # Zmniejsz tę wartość, jeśli masz problemy z pamięcią
    epochs = 5  # Zwiększ tę wartość dla lepszych wyników
    
    # Tworzenie data loaderów
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2  # Dostosuj do liczby rdzeni CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Sprawdzenie dostępności GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie: {device}")
    
    # Trenowanie modelu
    model, history = train_model(
        train_loader,
        val_loader,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        output_dir=models_dir
    )
    
    print("\nTrening zakończony!")

if __name__ == "__main__":
    main()
