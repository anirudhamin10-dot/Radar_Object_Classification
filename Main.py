"""
Pattern Recognition Project: Radar Scenes Multi-Category Classification
Dataset: Radar Scenes (https://radar-scenes.com/)
Method: Classical Pattern Recognition with Feature Extraction
Comparison: SVM vs Random Forest

"""

import numpy as np
import json
import os
from pathlib import Path
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import resample
import gc

def Balance_dataset(X, y, maxsamples=5000):
    #Balance dataset by downsampling classes with more than maxsamples

    uclass = np.unique(y)
    xBal = []
    yBal = []
        
    for cls in uclass:
        clsi = np.where(y == cls)[0]
        ogcount = len(clsi)
        
        if len(clsi) > maxsamples:
            clsi = resample(clsi, n_samples=maxsamples, 
                                  replace=False, random_state=42)
        
        xBal.append(X[clsi])
        yBal.append(y[clsi])
        print(f"Class '{cls}': original samples = {ogcount}, balanced samples = {len(clsi)}") 

    return np.vstack(xBal), np.hstack(yBal)

def Load_dataset(path='data', maxsamples=10000):
    # Load dataset from HDF5 files

    print("Loading Radar scenes dataset...")
    
    # Label mapping from RadarScenes readme

    Label = {
        0: 'car',
        1: 'large_vehicle',
        2: 'truck',
        3: 'bus',
        4: 'train',
        5: 'bicycle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'pedestrian_group',
        9: 'animal',
        10: 'other_dynamic',
        11: 'static'
    }
    
    # Track samples per class to stop early when limts reached
    classcount = {label: 0 for label in Label.values()}
    classfeatures = {label: [] for label in Label.values()}
    
    data_dir = Path(path)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{path}' not found!")
    
    sequence_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not sequence_dirs:
        raise FileNotFoundError(f"No sequence directories found in '{path}'!")
    
    print(f"Found {len(sequence_dirs)} sequences")
    print(f"Loading max {maxsamples} samples per class...")
    
    # Check if all classes are full

    def all_classes_full():
        return all(count >= maxsamples for count in classcount.values())
    
    for i, seq_dir in enumerate(sequence_dirs):
        if all_classes_full():
            print(f"  All classes reached maximum samples.")
            break
            
        file = seq_dir / 'radar_data.h5'
        
        if not file.exists():
            continue
        
        # Load radar data from HDF5 in parts

        with h5py.File(file, 'r') as h5f:
            dataset = h5f['radar_data']
            size = 5000
            
            for partStart in range(0, len(dataset), size):
                partEnd = min(partStart + size, len(dataset))
                dataParts = dataset[partStart:partEnd]
                
                # Process part

                for detect in dataParts:
                    lid = int(detect['label_id'])
                    
                    if lid not in Label:
                        continue
                    
                    lablename = Label[lid]
                    
                    # Skip if class already has enough samples

                    if classcount[lablename] >= maxsamples:
                        continue
                    
                    x_cc = float(detect['x_cc'])
                    y_cc = float(detect['y_cc'])
                    vr = float(detect['vr'])
                    vr_comp = float(detect['vr_compensated'])
                    rcs = float(detect['rcs'])
                    
                    features = Feature_extract(x_cc, y_cc, vr, vr_comp, rcs)
                    
                    classfeatures[lablename].append(features)
                    classcount[lablename] += 1
                
                # Delete part to free memory

                del dataParts
        
        if (i + 1) % 20 == 0:
            Total = sum(classcount.values())
            print(f"  Processed {i + 1}/{len(sequence_dirs)} sequences, {Total} samples loaded")
    
    # Convert to arrays

    featurelist = []
    lablelist = []
    
    print("\nFinal class distribution:")
    for lablename in sorted(classfeatures.keys()):
        count = len(classfeatures[lablename])
        if count > 0:
            featurelist.extend(classfeatures[lablename])
            lablelist.extend([lablename] * count)
            print(f"  {lablename:<20s}: {count:>6d} samples")
    
    # Clearing storage

    del classfeatures
    del classcount
    
    if len(featurelist) == 0:
        raise ValueError("No labeled samples found!")
    
    print(f"\nTotal loaded: {len(featurelist)} samples")
    return np.array(featurelist, dtype=np.float32), np.array(lablelist)

def Feature_extract(x, y, vr, vr_comp, rcs):

    # Feature Extraction: Transform raw radar measurements into discriminative features

    range_val = np.sqrt(x**2 + y**2)
    azimuth = np.arctan2(y, x)
    vel_diff = abs(vr - vr_comp)
    rcs_nomal = rcs / (range_val + 1e-6)
    spatial_vel = range_val * abs(vr)
    azimuth_vel = azimuth * vr
    log_rcs = np.log(abs(rcs) + 1)
    range_sq = range_val ** 2
    doppler_sig = vr * rcs
    compensated_sig = vr_comp * rcs
    
    return [
        x, y, vr, vr_comp, rcs,
        range_val, azimuth, vel_diff, rcs_nomal,
        spatial_vel, azimuth_vel,
        log_rcs, range_sq, doppler_sig, compensated_sig
    ]

def train_classifier(clf, clfname, X_train, y_train, X_test, y_test):
    """Train and evaluate a classifier"""
    print(f"Training {clfname}...")
    clf.fit(X_train, y_train)
    print(f"{clfname} training completed!")
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    # Save metrics to file
    with open(f"results_{clfname.lower().replace(' ', '_')}.txt", 'w') as f:
        f.write(f"{clfname} Results:\n")
        f.write(f"Accuracy:  {accuracy*100:.2f}%\n")
        f.write(f"Precision: {precision*100:.2f}%\n")
        f.write(f"Recall:    {recall*100:.2f}%\n")
        f.write(f"F1-Score:  {f1*100:.2f}%\n")
    
    return {
        'name': clfname,
        'classifier': clf,
        'predictions': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(y_test, y_pred, classes, clf_name, filename):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {clf_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {filename}")

def plot_comparison(results, filename):
    """Plot performance comparison between classifiers"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    svm_scores = [results[0]['accuracy'], results[0]['precision'], results[0]['recall'], results[0]['f1']]
    rf_scores = [results[1]['accuracy'], results[1]['precision'], results[1]['recall'], results[1]['f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, [score * 100 for score in svm_scores], width, label='SVM', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, [score * 100 for score in rf_scores], width, label='Random Forest', color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('SVM vs Random Forest Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 100.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPerformance comparison saved: {filename}")

def main():
    """Main execution pipeline"""
    # Load dataset (now with built-in balancing)
    X, y = Load_dataset('data', maxsamples=10000)
    
    # Force garbage collection
    gc.collect()
    
    print("No of features:", X.shape[1])
    print("No of classes with labels:", len(np.unique(y)), np.unique(y))
    print("No of instances:", len(X))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Feature standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("CLASSIFIER TRAINING AND EVALUATION")
    print("="*60)
    
    # Initialize classifiers with class balancing
    classifiers = [
        (SVC(kernel='rbf', class_weight='balanced', random_state=42), "SVM"),
        (RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, 
                                min_samples_leaf=2, class_weight='balanced', 
                                random_state=42, n_jobs=-1), "Random Forest")
    ]
    
    results = []
    classes = np.unique(y)
    
    # Train and evaluate each classifier
    for clf, clfname in classifiers:
        result = train_classifier(clf, clfname, X_train_scaled, y_train, X_test_scaled, y_test)
        results.append(result)
        
        # Plot confusion matrix
        cm_filename = f"confusion_matrix_{clfname.replace(' ', '_').lower()}.png"
        plot_confusion_matrix(y_test, result['predictions'], classes, clfname, cm_filename)
    
    # Save detailed classification reports to file
    with open("classification_reports.txt", "w") as f:
        f.write("DETAILED CLASSIFICATION REPORTS\n")
        f.write("="*60 + "\n\n")
        for result in results:
            f.write(f"{result['name']}:\n")
            f.write(classification_report(y_test, result['predictions'], zero_division=0))
            f.write("\n")
    
    # Plot comparison
    plot_comparison(results, "classifier_comparison.png")
    
    # Feature importance for Random Forest
    rfclf = results[1]['classifier']
    featurenames = ['x', 'y', 'vr', 'vr_comp', 'rcs', 'range', 'azimuth', 
                     'vel_diff', 'rcs_norm', 'spatial_vel', 'azimuth_vel', 
                     'log_rcs', 'range_sq', 'doppler_sig', 'comp_sig']
    
    importances = rfclf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Save feature importance analysis
    with open("feature_importance.txt", 'w') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS (Random Forest)\n\n")
        f.write("Top 10 Most Important Features:\n")
        for i in range(min(10, len(featurenames))):
            idx = indices[i]
            f.write(f"{i+1}. {featurenames[idx]:<15s}: {importances[idx]*100:.2f}%\n")
    
    best_clf = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest classifier: {best_clf['name']}")
    print(f"Best accuracy: {best_clf['accuracy']*100:.2f}%")
    print("Results and plots saved")

if __name__ == "__main__":
    main()