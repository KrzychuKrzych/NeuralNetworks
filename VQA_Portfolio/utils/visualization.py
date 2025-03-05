import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Dodanie ścieżki głównego katalogu projektu do sys.path
sys.path.append("VQA_Portfolio")

# Importowanie własnych modułów
from model.vqa_model import VQAModel, VQADataset, get_transforms

def analyze_dataset_statistics(csv_path, output_dir=None):
    """
    Analizuje i wizualizuje statystyki zbioru danych.
    
    Args:
        csv_path (str): Ścieżka do pliku CSV ze zbiorem danych
        output_dir (str): Katalog wyjściowy (opcjonalny)
    """
    # Wczytywanie danych
    df = pd.read_csv(csv_path)
    print(f"Wczytano {len(df)} rekordów ze zbioru danych.")
    
    # Utworzenie katalogu na wykresy
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # --- Analiza pytań ---
    
    # Długość pytań
    df['question_length'] = df['question'].apply(len)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['question_length'], bins=30, kde=True)
    plt.title('Rozkład Długości Pytań')
    plt.xlabel('Długość pytania (liczba znaków)')
    plt.ylabel('Częstość')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'question_length_distribution.png'))
        plt.close()
    else:
        plt.show()
    
    # Typy pytań (jeśli kolumna istnieje lub można utworzyć)
    if 'question_type' not in df.columns:
        # Prosta heurystyka do określania typu pytania
        def get_question_type(question):
            question = question.lower()
            if question.startswith('what'):
                return 'what'
            elif question.startswith('how'):
                return 'how'
            elif question.startswith('is'):
                return 'yes/no'
            elif question.startswith('are'):
                return 'yes/no'
            elif question.startswith('does'):
                return 'yes/no'
            elif question.startswith('do'):
                return 'yes/no'
            elif question.startswith('where'):
                return 'where'
            elif question.startswith('when'):
                return 'when'
            elif question.startswith('who'):
                return 'who'
            elif question.startswith('which'):
                return 'which'
            else:
                return 'other'
        
        df['question_type'] = df['question'].apply(get_question_type)
    
    # Rozkład typów pytań
    question_type_counts = df['question_type'].value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=question_type_counts.index, y=question_type_counts.values)
    plt.title('Rozkład Typów Pytań')
    plt.xlabel('Typ pytania')
    plt.ylabel('Liczba pytań')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'question_type_distribution.png'))
        plt.close()
    else:
        plt.show()
    
    # --- Analiza odpowiedzi (jeśli dostępne) ---
    
    if 'multiple_choice_answer' in df.columns:
        # Najczęstsze odpowiedzi
        answer_counts = df['multiple_choice_answer'].value_counts().head(20)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=answer_counts.index, y=answer_counts.values)
        plt.title('Top 20 Najczęstszych Odpowiedzi')
        plt.xlabel('Odpowiedź')
        plt.ylabel('Liczba wystąpień')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'top_answers_distribution.png'))
            plt.close()
        else:
            plt.show()
        
        # Długość odpowiedzi
        df['answer_length'] = df['multiple_choice_answer'].apply(len)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df['answer_length'], bins=30, kde=True)
        plt.title('Rozkład Długości Odpowiedzi')
        plt.xlabel('Długość odpowiedzi (liczba znaków)')
        plt.ylabel('Częstość')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'answer_length_distribution.png'))
            plt.close()
        else:
            plt.show()

def visualize_model_performance(history_path, output_dir=None):
    """
    Wizualizuje wydajność modelu na podstawie historii treningu.
    
    Args:
        history_path (str): Ścieżka do pliku CSV z historią treningu
        output_dir (str): Katalog wyjściowy (opcjonalny)
    """
    # Wczytywanie historii treningu
    history_df = pd.read_csv(history_path)
    
    # Utworzenie katalogu na wykresy
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Wykres straty
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Wykres dokładności
    plt.subplot(1, 2, 2)
    plt.plot(history_df['train_acc'], label='Train Acc')
    plt.plot(history_df['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_performance.png'))
        plt.close()
    else:
        plt.show()
    
    # Wykres czasu trwania epok
    if 'epoch_time' in history_df.columns:
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, len(history_df) + 1), history_df['epoch_time'])
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Time per Epoch')
        plt.xticks(range(1, len(history_df) + 1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'epoch_time.png'))
            plt.close()
        else:
            plt.show()

def visualize_feature_space(model, dataloader, device, n_samples=500, output_dir=None):
    """
    Wizualizuje przestrzeń cech wygenerowaną przez model przy użyciu t-SNE.
    
    Args:
        model: Wytrenowany model
        dataloader: DataLoader z danymi
        device: Urządzenie (CPU/GPU)
        n_samples (int): Liczba próbek do wizualizacji
        output_dir (str): Katalog wyjściowy (opcjonalny)
    """
    model.eval()
    
    # Kolekcjonowanie cech i etykiet
    image_features = []
    text_features = []
    labels = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= n_samples:
                break
            
            # Przeniesienie danych na urządzenie
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Ekstrakcja cech obrazów
            img_feats = model.image_model(images)
            img_feats = img_feats.view(img_feats.size(0), -1)
            
            # Ekstrakcja cech tekstu
            text_outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            txt_feats = text_outputs.pooler_output
            
            # Zapisanie cech i etykiet
            image_features.append(img_feats.cpu().numpy())
            text_features.append(txt_feats.cpu().numpy())
            
            if 'answer_idx' in batch:
                labels.append(batch['answer_idx'].cpu().numpy())
            
            sample_count += images.size(0)
    
    # Konwersja list na tablice NumPy
    image_features = np.vstack(image_features)[:n_samples]
    text_features = np.vstack(text_features)[:n_samples]
    combined_features = np.hstack([image_features, text_features])[:n_samples]
    
    if labels:
        labels = np.concatenate(labels)[:n_samples]
    
    # Redukcja wymiarowości za pomocą t-SNE dla cech obrazów
    print("Redukcja wymiarowości cech obrazów za pomocą t-SNE...")
    tsne_img = TSNE(n_components=2, random_state=42)
    img_features_tsne = tsne_img.fit_transform(image_features)
    
    # Redukcja wymiarowości za pomocą t-SNE dla cech tekstu
    print("Redukcja wymiarowości cech tekstu za pomocą t-SNE...")
    tsne_txt = TSNE(n_components=2, random_state=42)
    txt_features_tsne = tsne_txt.fit_transform(text_features)
    
    # Redukcja wymiarowości za pomocą t-SNE dla połączonych cech
    print("Redukcja wymiarowości połączonych cech za pomocą t-SNE...")
    tsne_combined = TSNE(n_components=2, random_state=42)
    combined_features_tsne = tsne_combined.fit_transform(combined_features)
    
    # Wizualizacja cech obrazów
    plt.figure(figsize=(12, 10))
    if len(labels) > 0:
        scatter = plt.scatter(img_features_tsne[:, 0], img_features_tsne[:, 1],
                            c=labels, cmap='tab20', alpha=0.7, s=30)
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(img_features_tsne[:, 0], img_features_tsne[:, 1], alpha=0.7, s=30)
    
    plt.title('t-SNE Visualization of Image Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'tsne_image_features.png'))
        plt.close()
    else:
        plt.show()
    
    # Wizualizacja cech tekstu
    plt.figure(figsize=(12, 10))
    if len(labels) > 0:
        scatter = plt.scatter(txt_features_tsne[:, 0], txt_features_tsne[:, 1],
                            c=labels, cmap='tab20', alpha=0.7, s=30)
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(txt_features_tsne[:, 0], txt_features_tsne[:, 1], alpha=0.7, s=30)
    
    plt.title('t-SNE Visualization of Text Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'tsne_text_features.png'))
        plt.close()
    else:
        plt.show()
    
    # Wizualizacja połączonych cech
    plt.figure(figsize=(12, 10))
    if len(labels) > 0:
        scatter = plt.scatter(combined_features_tsne[:, 0], combined_features_tsne[:, 1],
                            c=labels, cmap='tab20', alpha=0.7, s=30)
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(combined_features_tsne[:, 0], combined_features_tsne[:, 1], alpha=0.7, s=30)
    
    plt.title('t-SNE Visualization of Combined Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'tsne_combined_features.png'))
        plt.close()
    else:
        plt.show()

def main():
    """Główna funkcja do wizualizacji danych i wyników"""
    
    # Ścieżki do folderów
    base_dir = "VQA_Portfolio"
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "data", "models")
    results_dir = os.path.join(base_dir, "data", "results")
    viz_dir = os.path.join(base_dir, "data", "visualizations")
    
    # Utworzenie katalogu na wizualizacje
    os.makedirs(viz_dir, exist_ok=True)
    
    # Analiza statystyk zbioru danych
    print("Analizowanie statystyk zbioru danych...")
    csv_path = os.path.join(processed_dir, "vqa_processed.csv")
    if os.path.exists(csv_path):
        analyze_dataset_statistics(csv_path, output_dir=os.path.join(viz_dir, 'dataset_stats'))
    else:
        print(f"Ostrzeżenie: Brak pliku {csv_path}. Pomijanie analizy zbioru danych.")
    
    # Wizualizacja wydajności modelu
    print("\nWizualizowanie wydajności modelu...")
    history_path = os.path.join(models_dir, "training_history.csv")
    if os.path.exists(history_path):
        visualize_model_performance(history_path, output_dir=os.path.join(viz_dir, 'performance'))
    else:
        print(f"Ostrzeżenie: Brak pliku {history_path}. Pomijanie wizualizacji wydajności modelu.")
    
    # Tworzenie dashboardu z wynikami
    print("\nTworzenie dashboardu z wynikami...")
    if os.path.exists(results_dir):
        create_dashboard(results_dir, output_dir=os.path.join(viz_dir, 'dashboard'))
    else:
        print(f"Ostrzeżenie: Brak katalogu {results_dir}. Pomijanie tworzenia dashboardu.")
    
    # Opcjonalna wizualizacja przestrzeni cech (wymaga wczytania modelu i danych)
    # visualize_feature_space() - To zadanie wymaga dużo zasobów, więc jest opcjonalne
    
    print(f"\nWizualizacje zostały wygenerowane w katalogu: {viz_dir}")
    
if __name__ == "__main__":
    main()
    """
    Tworzy interaktywny dashboard z wynikami modelu.
    
    Args:
        results_dir (str): Katalog z wynikami
        output_dir (str): Katalog wyjściowy (opcjonalny)
    """
    # W lokalnym środowisku nie możemy tworzyć interaktywnego dashboardu,
    # ale możemy wygenerować zestaw wykresów i statystyk
    
    # Ustalenie katalogu wyjściowego
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'dashboard')
    os.makedirs(output_dir, exist_ok=True)
    
    # Wczytywanie wyników
    metrics_path = os.path.join(results_dir, 'metrics.json')
    results_path = os.path.join(results_dir, 'evaluation_results.csv')
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Tworzenie wykresu metryk
        plt.figure(figsize=(10, 6))
        metrics_values = [metrics.get('accuracy', 0), metrics.get('precision', 0),
                        metrics.get('recall', 0), metrics.get('f1', 0)]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        bars = plt.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        plt.ylim(0, 1.0)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Model Performance Metrics')
        
        # Dodawanie etykiet z wartościami
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 punkty nad słupkiem
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
        plt.close()
    
    # Analiza wyników predykcji
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        
        if 'correct_prediction' in results_df.columns:
            # Wizualizacja dokładności
            accuracy = results_df['correct_prediction'].mean()
            
            plt.figure(figsize=(8, 8))
            plt.pie([accuracy, 1-accuracy], labels=['Correct', 'Incorrect'],
                   colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90)
            plt.title('Prediction Accuracy')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'accuracy_pie.png'))
            plt.close()
            
            # Dokładność według typu pytania (jeśli dostępne)
            if 'question_type' in results_df.columns:
                type_acc = results_df.groupby('question_type')['correct_prediction'].mean().sort_values()
                
                plt.figure(figsize=(12, 6))
                sns.barplot(x=type_acc.index, y=type_acc.values)
                plt.title('Accuracy by Question Type')
                plt.xlabel('Question Type')
                plt.ylabel('Accuracy')
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'accuracy_by_type.png'))
                plt.close()
    
    # Tworzenie pliku HTML z podsumowaniem (prosty "dashboard")
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VQA Model Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1, h2 { color: #2c3e50; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
            img { max-width: 100%; height: auto; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Visual Question Answering Model Dashboard</h1>
        
        <div class="card">
            <h2>Performance Metrics</h2>
            <img src="performance_metrics.png" alt="Performance Metrics">
        </div>
        
        <div class="grid">
    """
    
    # Dodawanie dostępnych wykresów
    for img_file in os.listdir(output_dir):
        if img_file.endswith('.png') and img_file != 'performance_metrics.png':
            img_name = img_file.replace('.png', '').replace('_', ' ').title()
            html_content += f"""
            <div class="card">
                <h2>{img_name}</h2>
                <img src="{img_file}" alt="{img_name}">
            </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'dashboard.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard został wygenerowany w {os.path.join(output_dir, 'dashboard.html')}")
