import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import time

# Dodanie ścieżki głównego katalogu projektu do sys.path
sys.path.append("VQA_Portfolio")

# Importowanie własnych modułów
from model.vqa_model import VQAModel, VQADataset, get_transforms

def load_trained_model(model_path, num_classes, device):
    """
    Wczytuje wytrenowany model.
    
    Args:
        model_path (str): Ścieżka do pliku z zapisanym modelem
        num_classes (int): Liczba klas (możliwych odpowiedzi)
        device (torch.device): Urządzenie (CPU/GPU)
    
    Returns:
        model: Wczytany model
    """
    # Inicjalizacja modelu
    model = VQAModel(num_classes=num_classes).to(device)
    
    # Wczytywanie wag
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Wczytano model z {model_path}")
    return model

def evaluate_model(model, dataloader, device, idx_to_answer=None):
    """
    Ewaluuje model na zbiorze testowym.
    
    Args:
        model: Model do ewaluacji
        dataloader: DataLoader z danymi testowymi
        device: Urządzenie (CPU/GPU)
        idx_to_answer: Opcjonalny słownik mapujący indeksy na odpowiedzi
    
    Returns:
        dict: Wyniki ewaluacji oraz predykcje
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_question_ids = []
    all_questions = []
    all_probs = []  # Prawdopodobieństwa dla każdej klasy
    
    print("Rozpoczęcie ewaluacji modelu...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Przeniesienie danych na urządzenie
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(images, input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            # Zapisanie predykcji
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_question_ids.extend(batch['question_id'].numpy())
            all_questions.extend(batch['question'])
            
            # Zapisanie etykiet, jeśli są dostępne
            if 'answer_idx' in batch:
                all_labels.extend(batch['answer_idx'].cpu().numpy())
            
            # Wyświetlanie postępu
            if (batch_idx + 1) % 10 == 0:
                print(f"\rPrzetworzono {batch_idx + 1}/{len(dataloader)} batchy...", end="")
    
    print(f"\nEwaluacja zajęła {time.time() - start_time:.2f} sekund")
    
    # Konwersja na numpy array
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_question_ids = np.array(all_question_ids)
    
    # Przygotowanie wyników
    results = {
        'predictions': all_preds,
        'probabilities': all_probs, 
        'question_ids': all_question_ids,
        'questions': all_questions
    }
    
    # Dodanie rzeczywistych etykiet, jeśli są dostępne
    if all_labels:
        all_labels = np.array(all_labels)
        results['ground_truth'] = all_labels
        
        # Obliczenie metryk
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Dodanie metryk do wyników
        results['metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # Konwersja indeksów odpowiedzi na rzeczywiste odpowiedzi
    if idx_to_answer and 'predictions' in results:
        pred_answers = []
        for pred in results['predictions']:
            pred_answers.append(idx_to_answer.get(str(pred), "unknown"))
        results['predicted_answers'] = pred_answers
        
        if 'ground_truth' in results:
            true_answers = []
            for label in results['ground_truth']:
                true_answers.append(idx_to_answer.get(str(label), "unknown"))
            results['true_answers'] = true_answers
    
    return results

def save_evaluation_results(results, output_dir):
    """
    Zapisuje wyniki ewaluacji do plików.
    
    Args:
        results (dict): Wyniki ewaluacji
        output_dir (str): Katalog wyjściowy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tworzenie DataFrame z wynikami
    df = pd.DataFrame({
        'question_id': results['question_ids'],
        'question': results['questions'],
        'predicted_idx': results['predictions']
    })
    
    # Dodawanie odpowiedzi tekstowych, jeśli są dostępne
    if 'predicted_answers' in results:
        df['predicted_answer'] = results['predicted_answers']
    
    # Dodawanie rzeczywistych etykiet, jeśli są dostępne
    if 'ground_truth' in results:
        df['true_idx'] = results['ground_truth']
        
    if 'true_answers' in results:
        df['true_answer'] = results['true_answers']
    
    # Dodawanie flagi dla poprawnych predykcji
    if 'ground_truth' in results:
        df['correct_prediction'] = df['predicted_idx'] == df['true_idx']
    
    # Zapisywanie DataFrame do pliku CSV
    df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    
    # Zapisywanie metryk, jeśli są dostępne
    if 'metrics' in results:
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(results['metrics'], f, indent=4)
    
    print(f"Zapisano wyniki ewaluacji do {output_dir}")

def plot_confusion_matrix(results, top_n_classes=20, output_dir=None):
    """
    Tworzy macierz pomyłek dla najczęstszych klas.
    
    Args:
        results (dict): Wyniki ewaluacji
        top_n_classes (int): Liczba najczęstszych klas do uwzględnienia
        output_dir (str): Katalog wyjściowy (opcjonalny)
    """
    if 'ground_truth' not in results or 'predictions' not in results:
        print("Brak etykiet rzeczywistych do utworzenia macierzy pomyłek.")
        return
    
    # Wybór najczęstszych klas
    unique_labels, counts = np.unique(results['ground_truth'], return_counts=True)
    top_indices = np.argsort(-counts)[:top_n_classes]
    top_classes = unique_labels[top_indices]
    
    # Filtrowanie wyników do uwzględnienia tylko najczęstszych klas
    mask = np.isin(results['ground_truth'], top_classes)
    filtered_true = results['ground_truth'][mask]
    filtered_pred = results['predictions'][mask]
    
    # Konwersja etykiet do nazw klas, jeśli dostępne
    if 'true_answers' in results and 'predicted_answers' in results:
        true_labels = [results['true_answers'][i] for i, m in enumerate(mask) if m]
        pred_labels = [results['predicted_answers'][i] for i, m in enumerate(mask) if m]
        label_names = np.unique(true_labels)
    else:
        true_labels = filtered_true
        pred_labels = filtered_pred
        label_names = [str(c) for c in top_classes]
    
    # Obliczenie macierzy pomyłek
    cm = confusion_matrix(true_labels, pred_labels, labels=label_names)
    
    # Normalizacja macierzy pomyłek
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Tworzenie wykresu
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
    else:
        plt.show()

def analyze_errors(results, output_dir=None):
    """
    Analizuje błędy modelu.
    
    Args:
        results (dict): Wyniki ewaluacji
        output_dir (str): Katalog wyjściowy (opcjonalny)
    """
    if 'ground_truth' not in results or 'predictions' not in results:
        print("Brak etykiet rzeczywistych do analizy błędów.")
        return
    
    # Tworzenie DataFrame z wynikami
    df = pd.DataFrame({
        'question_id': results['question_ids'],
        'question': results['questions'],
        'true_idx': results['ground_truth'],
        'predicted_idx': results['predictions'],
        'correct': results['ground_truth'] == results['predictions']
    })
    
    if 'true_answers' in results and 'predicted_answers' in results:
        df['true_answer'] = results['true_answers']
        df['predicted_answer'] = results['predicted_answers']
    
    # Dodawanie typu pytania (prosta heurystyka)
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
    
    # Analiza dokładności według typu pytania
    question_type_accuracy = df.groupby('question_type')['correct'].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    question_type_accuracy.plot(kind='bar')
    plt.title('Accuracy by Question Type')
    plt.xlabel('Question Type')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'accuracy_by_question_type.png'))
        plt.close()
    else:
        plt.show()
    
    # Analiza najczęstszych błędów
    error_df = df[~df['correct']]
    
    if len(error_df) > 0:
        # Najczęstsze typy pytań z błędami
        error_types = error_df['question_type'].value_counts()
        
        plt.figure(figsize=(10, 6))
        error_types.plot(kind='bar')
        plt.title('Most Common Question Types with Errors')
        plt.xlabel('Question Type')
        plt.ylabel('Error Count')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'error_question_types.png'))
            plt.close()
        else:
            plt.show()
        
        # Zapisywanie najczęstszych błędów do pliku
        if output_dir:
            error_df.to_csv(os.path.join(output_dir, 'error_analysis.csv'), index=False)
            print(f"Zapisano analizę błędów do {os.path.join(output_dir, 'error_analysis.csv')}")

def main():
    """Główna funkcja do ewaluacji modelu"""
    
    # Ścieżki do folderów
    base_dir = "VQA_Portfolio"
    processed_dir = os.path.join(base_dir, "data", "processed")
    images_dir = os.path.join(base_dir, "data", "raw", "train2014")
    models_dir = os.path.join(base_dir, "data", "models")
    results_dir = os.path.join(base_dir, "data", "results")
    
    # Sprawdzenie dostępności GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie: {device}")
    
    # Wczytywanie mapowania odpowiedzi
    answer_map_path = os.path.join(processed_dir, "idx_to_answer.json")
    if os.path.exists(answer_map_path):
        with open(answer_map_path, 'r') as f:
            idx_to_answer = json.load(f)
        print(f"Wczytano mapowanie dla {len(idx_to_answer)} odpowiedzi.")
    else:
        idx_to_answer = None
        print("Ostrzeżenie: Brak pliku z mapowaniem odpowiedzi.")
    
    # Wczytywanie modelu
    model_path = os.path.join(models_dir, "vqa_model_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(models_dir, "vqa_model_last.pth")
        if not os.path.exists(model_path):
            print("Błąd: Brak pliku modelu. Najpierw uruchom training.py.")
            return
    
    # Wczytywanie konfiguracji treningu, aby uzyskać liczbę klas
    config_path = os.path.join(models_dir, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        num_classes = config.get('num_classes', 1001)
    else:
        num_classes = 1001  # Domyślna wartość
        print("Ostrzeżenie: Brak pliku konfiguracji treningu. Przyjęto domyślną liczbę klas.")
    
    # Wczytywanie modelu
    model = load_trained_model(model_path, num_classes, device)
    
    # Wczytywanie zbioru testowego
    test_csv = os.path.join(processed_dir, "val.csv")  # Używamy zbioru walidacyjnego jako testowego
    if not os.path.exists(test_csv):
        print(f"Błąd: Brak pliku ze zbiorem testowym {test_csv}.")
        return
    
    # Transformacje dla obrazów
    _, test_transform = get_transforms()
    
    # Tworzenie datasetu testowego
    test_dataset = VQADataset(
        test_csv,
        images_dir,
        transform=test_transform,
        answer_map_path=os.path.join(processed_dir, "answer_to_idx.json")
    )
    
    # Tworzenie data loadera
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    # Ewaluacja modelu
    results = evaluate_model(model, test_loader, device, idx_to_answer)
    
    # Tworzenie katalogu na wyniki
    os.makedirs(results_dir, exist_ok=True)
    
    # Zapisywanie wyników
    save_evaluation_results(results, results_dir)
    
    # Wizualizacje i analizy
    if 'ground_truth' in results:
        # Macierz pomyłek
        plot_confusion_matrix(results, top_n_classes=10, output_dir=results_dir)
        
        # Analiza błędów
        analyze_errors(results, output_dir=results_dir)
    
    # Wizualizacja przykładów
    visualize_examples(results, images_dir, num_examples=10, output_dir=results_dir)
    
    print("\nEwaluacja zakończona! Wyniki zapisano do:", results_dir)
    
if __name__ == "__main__":
    main()

def visualize_examples(results, img_dir, num_examples=5, output_dir=None, random_seed=42):
    """
    Wizualizuje przykładowe predykcje modelu.
    
    Args:
        results (dict): Wyniki ewaluacji
        img_dir (str): Katalog z obrazami
        num_examples (int): Liczba przykładów do wizualizacji
        output_dir (str): Katalog wyjściowy (opcjonalny)
        random_seed (int): Ziarno losowości
    """
    """
    Wizualizuje przykładowe predykcje modelu.
    
    Args:
        results (dict): Wyniki ewaluacji
        img_dir (str): Katalog z obrazami
        num_examples (int): Liczba przykładów do wizualizacji
        output_dir (str): Katalog wyjściowy (opcjonalny)
        random_seed (int): Ziarno losowości
    """
    # Ustawienie ziarna losowości dla powtarzalności
    random.seed(random_seed)
    
    # Sprawdzenie, czy mamy dostęp do prawdziwych etykiet
    has_ground_truth = 'ground_truth' in results
    
    # Wczytanie dataframe z wynikami
    df = pd.DataFrame({
        'question_id': results['question_ids'],
        'question': results['questions'],
        'predicted_idx': results['predictions']
    })
    
    if 'predicted_answers' in results:
        df['predicted_answer'] = results['predicted_answers']
    
    if has_ground_truth:
        df['true_idx'] = results['ground_truth']
        
    if 'true_answers' in results:
        df['true_answer'] = results['true_answers']
        
    if has_ground_truth:
        df['correct'] = df['predicted_idx'] == df['true_idx']
        
        # Wizualizacja zarówno poprawnych, jak i niepoprawnych predykcji
        correct_samples = df[df['correct']].sample(min(num_examples // 2, sum(df['correct'])), random_state=random_seed)
        incorrect_samples = df[~df['correct']].sample(min(num_examples - len(correct_samples), sum(~df['correct'])), random_state=random_seed)
        samples = pd.concat([correct_samples, incorrect_samples])
    else:
        # Jeśli brak etykiet rzeczywistych, wybieramy losowe przykłady
        samples = df.sample(num_examples, random_state=random_seed)
    
    # Przygotowanie folderu na wizualizacje
    if output_dir:
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    
    # Wizualizacja przykładów
    for i, (_, sample) in enumerate(samples.iterrows()):
        img_id = int(sample['question_id'].split('_')[-1])  # Zakładamy format COCO_train2014_000000123456
        question = sample['question']
        pred_answer = sample['predicted_answer'] if 'predicted_answer' in sample else f"Class {sample['predicted_idx']}"
        
        # Ścieżka do obrazu
        img_path = os.path.join(img_dir, f"COCO_train2014_{img_id:012d}.jpg")
        
        try:
            # Wczytywanie obrazu
            image = Image.open(img_path).convert('RGB')
            
            # Tworzenie wykresu
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.axis('off')
            
            # Dodawanie informacji
            title = f"Pytanie: {question}\nPredykcja: {pred_answer}"
            
            if has_ground_truth and 'true_answer' in sample:
                title += f"\nPrawdziwa odpowiedź: {sample['true_answer']}"
                correct = sample['correct'] if 'correct' in sample else (sample['predicted_idx'] == sample['true_idx'])
                title += f"\nStatus: {'✓ Poprawna' if correct else '✗ Niepoprawna'}"
            
            plt.title(title, fontsize=12)
            
            # Zapisywanie lub wyświetlanie
            if output_dir:
                plt.savefig(os.path.join(viz_dir, f"example_{i+1}.png"), bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f"Błąd podczas wizualizacji obrazu {img_path}: {e}")
