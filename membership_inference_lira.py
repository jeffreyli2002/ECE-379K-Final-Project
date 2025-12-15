
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import os
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs('outputs_lira', exist_ok=True)

# Load the diabetes hospital readmission dataset
# Should have ~100k records from 1999-2008
# Task: Predict 30-day hospital readmission
def load_diabetes_dataset(data_dir='Diabetic_Database'):
    try:
        print(f"   Loading dataset from: {data_dir}")
        
        data = pd.read_csv(os.path.join(data_dir, 'diabetic_data.csv'))
        print(f"   Loaded {len(data)} encounters")
        
        # Target: readmitted within 30 days (binary)
        data['readmitted_binary'] = (data['readmitted'] == '<30').astype(int)
        
        # Convert age ranges to numeric
        age_map = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        data['age_numeric'] = data['age'].map(age_map)
        
        data['gender_male'] = (data['gender'] == 'Male').astype(int)
        
        # Race encoding
        data['race_caucasian'] = (data['race'] == 'Caucasian').astype(int)
        data['race_african_american'] = (data['race'] == 'AfricanAmerican').astype(int)
        
        # Clinical features
        clinical_features = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses'
        ]
        
        data['change_made'] = (data['change'] == 'Ch').astype(int)
        data['diabetes_med'] = (data['diabetesMed'] == 'Yes').astype(int)
        data['a1c_tested'] = (data['A1Cresult'] != 'None').astype(int)
        data['glucose_tested'] = (data['max_glu_serum'] != 'None').astype(int)
        
        # Final feature set
        feature_cols = (
            ['age_numeric', 'gender_male', 'race_caucasian', 'race_african_american'] +
            clinical_features +
            ['change_made', 'diabetes_med', 'a1c_tested', 'glucose_tested']
        )
        
        data = data.dropna(subset=['age_numeric'] + clinical_features)
        
        if len(data) < 1000:
            raise ValueError(f"Insufficient data: only {len(data)} samples after preprocessing")
        
        X = data[feature_cols].values
        y = data['readmitted_binary'].values
        X = np.nan_to_num(X, nan=0.0)
        
        print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Readmission rate: {y.mean():.2%}")
        
        # Save for later
        output_df = pd.DataFrame(X, columns=feature_cols)
        output_df['readmitted_30day'] = y
        output_df.to_csv('diabetes_processed_dataset.csv', index=False)
        print(f"   Saved to diabetes_processed_dataset.csv")
        
        return X, y
        
    except Exception as e:
        print(f"   Error loading diabetes dataset: {e}")
        raise


# LiRA implementation. Trains shadow models with/without samples,
# fits Gaussians to loss distributions, computes likelihood ratios.
# For each target sample:
#   1. Train N shadow models WITH it and M shadow models WITHOUT it
#   2. Collect loss distributions: P(loss | IN) and P(loss | OUT)
#   3. For target model's loss on sample, compute likelihood ratio
class LiRA_Attack:
    
    def __init__(self, model_class, model_kwargs, num_shadow_models=16):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.num_shadow_models = num_shadow_models
        self.loss_distributions = {}  # {idx: {'in': [...], 'out': [...]}}}
        
    # Compute cross-entropy loss for samples
    def _compute_loss(self, model, X, y):
        probs = model.predict_proba(X)
        true_class_probs = probs[np.arange(len(y)), y]
        losses = -np.log(true_class_probs + 1e-12)
        return losses
    
    # Train shadow models with/without each target sample
    # Args: X_train/y_train (pool), X_target/y_target (samples to test), target_indices (track these)
    def train_shadow_models(self, X_train, y_train, X_target, y_target, target_indices):
        print(f"\nTraining {self.num_shadow_models * 2} shadow models")
        print(f"Tracking membership for {len(target_indices)} samples")
        
        # Initialize loss storage
        for idx in target_indices:
            self.loss_distributions[idx] = {'in': [], 'out': []}
        
        # Train shadow models
        total_models = self.num_shadow_models * 2
        
        with tqdm(total=total_models, desc="Shadow training") as pbar:
            for i in range(self.num_shadow_models):
                # Model WITH target samples (IN)
                shadow_model_in = self.model_class(**self.model_kwargs)
                
                # Randomly sample training data + include target samples
                sample_size = len(X_train) // 2
                random_indices = np.random.choice(len(X_train), sample_size, replace=False)
                
                # Combine random samples with target samples
                X_shadow_in = np.vstack([X_train[random_indices], X_target])
                y_shadow_in = np.concatenate([y_train[random_indices], y_target])
                
                shadow_model_in.fit(X_shadow_in, y_shadow_in)
                
                # Compute losses for target samples
                losses_in = self._compute_loss(shadow_model_in, X_target, y_target)
                for j, idx in enumerate(target_indices):
                    self.loss_distributions[idx]['in'].append(losses_in[j])
                
                pbar.update(1)
                
                # Model WITHOUT target samples (OUT)
                shadow_model_out = self.model_class(**self.model_kwargs)
                
                # Use same random sample but WITHOUT target samples
                X_shadow_out = X_train[random_indices]
                y_shadow_out = y_train[random_indices]
                
                shadow_model_out.fit(X_shadow_out, y_shadow_out)
                
                # Compute losses for target samples
                losses_out = self._compute_loss(shadow_model_out, X_target, y_target)
                for j, idx in enumerate(target_indices):
                    self.loss_distributions[idx]['out'].append(losses_out[j])
                
                pbar.update(1)
        
        print("Shadow model training complete!")
    
    # Fit Gaussians to IN/OUT loss distributions
    # For each sample: N(μ_in, σ_in) for losses when IN, N(μ_out, σ_out) when OUT
    def fit_gaussian_distributions(self):
        self.gaussian_params = {}
        
        for idx in self.loss_distributions.keys():
            losses_in = np.array(self.loss_distributions[idx]['in'])
            losses_out = np.array(self.loss_distributions[idx]['out'])
            
            # Fit Gaussians
            mu_in, sigma_in = losses_in.mean(), losses_in.std() + 1e-6
            mu_out, sigma_out = losses_out.mean(), losses_out.std() + 1e-6
            
            self.gaussian_params[idx] = {
                'in': (mu_in, sigma_in),
                'out': (mu_out, sigma_out)
            }
    
    # LR = P(loss|IN) / P(loss|OUT)
    # Higher LR → more likely the sample was IN the training set
    def compute_likelihood_ratio(self, target_loss, sample_idx):
        if sample_idx not in self.gaussian_params:
            return 1.0
        
        mu_in, sigma_in = self.gaussian_params[sample_idx]['in']
        mu_out, sigma_out = self.gaussian_params[sample_idx]['out']
        
        p_in = stats.norm.pdf(target_loss, mu_in, sigma_in)
        p_out = stats.norm.pdf(target_loss, mu_out, sigma_out)
        
        return (p_in + 1e-12) / (p_out + 1e-12)
    
    def compute_confidence(self, target_loss, sample_idx):
        if sample_idx not in self.gaussian_params:
            return 0.5
        
        mu_in, sigma_in = self.gaussian_params[sample_idx]['in']
        mu_out, sigma_out = self.gaussian_params[sample_idx]['out']
        
        p_in = stats.norm.pdf(target_loss, mu_in, sigma_in)
        p_out = stats.norm.pdf(target_loss, mu_out, sigma_out)
        
        return p_in / (p_in + p_out + 1e-12)
    
    # Run the attack on target model
    # Returns: scores (likelihood ratios), confidences (calibrated probabilities)
    def attack(self, target_model, X_test, y_test, test_indices):
        test_losses = self._compute_loss(target_model, X_test, y_test)
        
        scores = []
        confidences = []
        
        for i, idx in enumerate(test_indices):
            lr = self.compute_likelihood_ratio(test_losses[i], idx)
            conf = self.compute_confidence(test_losses[i], idx)
            
            scores.append(lr)
            confidences.append(conf)
        
        return np.array(scores), np.array(confidences)

def compute_metrics(scores, true_labels):
    # Use sklearn for ROC curve and AUC
    fpr_list, tpr_list, _ = roc_curve(true_labels, scores)
    auc = roc_auc_score(true_labels, scores)
    
    # Find TPR at specific FPR thresholds
    tpr_at_01 = 0
    tpr_at_001 = 0
    for i in range(len(fpr_list)):
        if fpr_list[i] <= 0.01:
            tpr_at_01 = max(tpr_at_01, tpr_list[i])
        if fpr_list[i] <= 0.001:
            tpr_at_001 = max(tpr_at_001, tpr_list[i])
    
    # Accuracy at median threshold
    threshold = np.median(scores)
    preds = (scores > threshold).astype(int)
    accuracy = (preds == true_labels).mean()
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'tpr_at_1%_fpr': tpr_at_01,
        'tpr_at_0.1%_fpr': tpr_at_001,
        'tpr_list': tpr_list,
        'fpr_list': fpr_list
    }

if __name__ == "__main__":
    print("LiRA Membership Inference Attack")
    print()
    
    NUM_SHADOW_MODELS = 16
    TRAIN_SIZE = 20000
    
    print(f"Config: {NUM_SHADOW_MODELS} shadow models, {TRAIN_SIZE} training samples")
    print()
    
    print("1. Loading data")
    X, y = load_diabetes_dataset('Diabetic_Database')
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    actual_train_size = min(TRAIN_SIZE, len(X) // 3)
    
    # Split into train/test/auxiliary
    X_temp, X_auxiliary, y_temp, y_auxiliary = train_test_split(
        X, y, test_size=0.33, random_state=RANDOM_SEED, stratify=y
    )
    
    if len(X_temp) < actual_train_size * 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
        )
    else:
        train_indices = np.random.choice(len(X_temp), actual_train_size, replace=False)
        test_indices = np.array([i for i in range(len(X_temp)) if i not in train_indices])
        
        X_train = X_temp[train_indices]
        y_train = y_temp[train_indices]
        X_test = X_temp[test_indices]
        y_test = y_temp[test_indices]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Aux: {len(X_auxiliary)}")
    print()
    
    models_config = {
        'Logistic Regression': {
            'class': LogisticRegression,
            'kwargs': {'max_iter': 1000, 'random_state': RANDOM_SEED}
        },
        'Random Forest': {
            'class': RandomForestClassifier,
            'kwargs': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 2,
                'random_state': RANDOM_SEED
            }
        }
    }
    
    if XGBOOST_AVAILABLE:
        models_config['XGBoost'] = {
            'class': XGBClassifier,
            'kwargs': {
                'n_estimators': 200,
                'max_depth': 10,
                'learning_rate': 0.1,
                'reg_alpha': 0,
                'reg_lambda': 0,
                'random_state': RANDOM_SEED,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
        }
    
    all_results = {}
    
    for model_name, config in models_config.items():
        print(f"\nATTACKING: {model_name}")
        
        # Train target model
        print("\n2. Training target model")
        target_model = config['class'](**config['kwargs'])
        target_model.fit(X_train, y_train)
        
        train_acc = target_model.score(X_train, y_train)
        test_acc = target_model.score(X_test, y_test)
        print(f"   Train accuracy: {train_acc:.4f}")
        print(f"   Test accuracy: {test_acc:.4f}")
        print(f"   Gap: {train_acc - test_acc:.4f}")
        
        # Select samples to track
        num_tracked_members = min(10000, len(X_train))
        num_tracked_nonmembers = min(10000, len(X_test))
        
        tracked_member_indices = np.random.choice(len(X_train), num_tracked_members, replace=False)
        tracked_nonmember_indices = np.random.choice(len(X_test), num_tracked_nonmembers, replace=False)
        
        X_tracked_members = X_train[tracked_member_indices]
        y_tracked_members = y_train[tracked_member_indices]
        X_tracked_nonmembers = X_test[tracked_nonmember_indices]
        y_tracked_nonmembers = y_test[tracked_nonmember_indices]
        
        # Combine for LiRA training
        X_tracked = np.vstack([X_tracked_members, X_tracked_nonmembers])
        y_tracked = np.concatenate([y_tracked_members, y_tracked_nonmembers])
        tracked_indices = list(range(len(X_tracked)))
        
        print("\n3. Training shadow models")
        lira = LiRA_Attack(
            model_class=config['class'],
            model_kwargs=config['kwargs'],
            num_shadow_models=NUM_SHADOW_MODELS
        )
        lira.train_shadow_models(
            X_auxiliary, y_auxiliary,
            X_tracked, y_tracked,
            tracked_indices
        )
        
        # Fit Gaussians
        print("\n4. Fitting Gaussians")
        lira.fit_gaussian_distributions()
        
        # Perform attack
        print("\n5. Running attack")
        scores, confidences = lira.attack(target_model, X_tracked, y_tracked, tracked_indices)
        
        # Evaluate
        true_labels = np.concatenate([
            np.ones(num_tracked_members),
            np.zeros(num_tracked_nonmembers)
        ])
        
        metrics = compute_metrics(scores, true_labels)
        
        print(f"\n6. Attack results:")
        print(f"   AUC: {metrics['auc']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   TPR @ 1% FPR: {metrics['tpr_at_1%_fpr']:.4f}")
        print(f"   TPR @ 0.1% FPR: {metrics['tpr_at_0.1%_fpr']:.4f}")
        
        all_results[model_name] = {
            **metrics,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': train_acc - test_acc
        }
        
        print()
    
    # Summary
    print("\nSUMMARY")
    print(f"{'Model':<20} | {'Gap':>6} | {'AUC':>6} | {'Acc':>6} | TPR@1%")
    print("-" * 70)
    for model_name, res in all_results.items():
        print(f"{model_name:<20} | {res['gap']:6.4f} | {res['auc']:6.4f} | "
              f"{res['accuracy']:6.4f} | {res['tpr_at_1%_fpr']:6.4f}")
    
    # Visualization
    print("\n7. Saving plots")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC curves
    colors = ['C0', 'C1', 'C2']
    for i, (model_name, res) in enumerate(all_results.items()):
        axes[0].plot(res['fpr_list'], res['tpr_list'],
                    label=f"{model_name}: {res['auc']:.3f}",
                    color=colors[i])
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves - LiRA Attack')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC comparison
    model_names = list(all_results.keys())
    aucs = [all_results[m]['auc'] for m in model_names]
    axes[1].bar(model_names, aucs, color=colors[:len(model_names)])
    axes[1].axhline(0.5, color='red', linestyle='--', label='Random')
    axes[1].set_ylabel('Attack AUC')
    axes[1].set_title('LiRA Attack Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('outputs_lira/lira_attack_results.png', dpi=150, bbox_inches='tight')
    print("   Saved: outputs_lira/lira_attack_results.png")
    
    print("\nDone!")
