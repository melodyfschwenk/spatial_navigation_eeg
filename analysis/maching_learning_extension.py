"""
Machine Learning Extension for Spatial Navigation EEG Analysis
==============================================================

This module extends the Spatial Navigation EEG Analysis Toolkit with machine learning capabilities
for advanced analysis of navigation strategies and neural patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import mne
from mne.decoding import CSP
from scipy import stats
import warnings


class EEGFeatureExtraction:
    """
    Class for extracting features from EEG data for machine learning.
    """
    
    def __init__(self, eeg_preprocessing):
        """
        Initialize the feature extraction.
        
        Parameters
        ----------
        eeg_preprocessing : EEGPreprocessing
            The EEG preprocessing object with loaded data
        """
        self.eeg = eeg_preprocessing
        self.features = {}
    
    def extract_band_power(self, participant_id, bands=None):
        """
        Extract frequency band power features from epoched EEG data.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
        bands : dict, optional
            Dictionary of frequency bands to extract, e.g., {'theta': (4, 8), 'alpha': (8, 13)}
            
        Returns
        -------
        numpy.ndarray
            Array of shape (n_epochs, n_channels * n_bands) containing band power features
        """
        if participant_id not in self.eeg.epochs:
            warnings.warn(f"No epochs data found for participant {participant_id}")
            return None
        
        # Default frequency bands if not specified
        if bands is None:
            bands = {
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
        
        epochs = self.eeg.epochs[participant_id]
        
        # Initialize feature array
        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_bands = len(bands)
        features = np.zeros((n_epochs, n_channels * n_bands))
        
        # Calculate band power for each frequency band and each channel
        for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            # Compute power spectral density
            psds, freqs = mne.time_frequency.psd_welch(
                epochs,
                fmin=fmin,
                fmax=fmax,
                n_fft=256,
                n_overlap=128,
                n_per_seg=256
            )
            
            # Average power over frequency band
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.mean(psds[:, :, freq_mask], axis=2)
            
            # Log-transform power to make it more normally distributed
            band_power = np.log10(band_power)
            
            # Store features (reshape to have one column per channel)
            features[:, i*n_channels:(i+1)*n_channels] = band_power
        
        # Store in features dictionary
        self.features[f"{participant_id}_band_power"] = features
        
        return features
    
    def extract_connectivity(self, participant_id, method='wpli', bands=None):
        """
        Extract connectivity features from epoched EEG data.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
        method : str
            The connectivity method to use ('coh', 'plv', 'wpli')
        bands : dict, optional
            Dictionary of frequency bands to extract
            
        Returns
        -------
        numpy.ndarray
            Array of shape (n_epochs, n_connections * n_bands) containing connectivity features
        """
        if participant_id not in self.eeg.epochs:
            warnings.warn(f"No epochs data found for participant {participant_id}")
            return None
        
        # Default frequency bands if not specified
        if bands is None:
            bands = {
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30)
            }
        
        epochs = self.eeg.epochs[participant_id]
        
        # Initialize feature array
        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_connections = (n_channels * (n_channels - 1)) // 2  # Number of unique connections
        n_bands = len(bands)
        features = np.zeros((n_epochs, n_connections * n_bands))
        
        # Calculate connectivity for each frequency band
        for band_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            # Compute connectivity
            try:
                con = mne.connectivity.spectral_connectivity(
                    epochs,
                    method=method,
                    mode='multitaper',
                    sfreq=epochs.info['sfreq'],
                    fmin=fmin,
                    fmax=fmax,
                    faverage=True,
                    mt_adaptive=True
                )
                
                # Reshape connectivity matrix to feature vector
                # (only take the lower triangle since the matrix is symmetric)
                con_matrix = con[0][:, :, 0]  # Get the first (and only) frequency bin
                tril_indices = np.tril_indices(n_channels, k=-1)
                con_vector = con_matrix[tril_indices]
                
                # Store in feature array
                features[:, band_idx*n_connections:(band_idx+1)*n_connections] = con_vector
            
            except Exception as e:
                warnings.warn(f"Error calculating connectivity for {band_name} band: {e}")
                continue
        
        # Store in features dictionary
        self.features[f"{participant_id}_connectivity_{method}"] = features
        
        return features
    
    def extract_time_domain_features(self, participant_id, time_windows=None):
        """
        Extract time domain features from epoched EEG data.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
        time_windows : list, optional
            List of time windows to extract features from, e.g., [(0, 0.2), (0.2, 0.4)]
            
        Returns
        -------
        numpy.ndarray
            Array of shape (n_epochs, n_channels * n_windows * n_features) containing time domain features
        """
        if participant_id not in self.eeg.epochs:
            warnings.warn(f"No epochs data found for participant {participant_id}")
            return None
        
        # Default time windows if not specified
        if time_windows is None:
            time_windows = [(0, 0.2), (0.2, 0.4), (0.4, 0.6)]
        
        epochs = self.eeg.epochs[participant_id]
        
        # Initialize feature array
        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_windows = len(time_windows)
        n_features = 4  # mean, std, min, max
        features = np.zeros((n_epochs, n_channels * n_windows * n_features))
        
        # Get data
        data = epochs.get_data()
        times = epochs.times
        
        # Extract features for each time window
        feature_idx = 0
        for window_idx, (tmin, tmax) in enumerate(time_windows):
            # Find time indices corresponding to the window
            time_mask = (times >= tmin) & (times <= tmax)
            
            # Extract window data
            window_data = data[:, :, time_mask]
            
            # Calculate features for each channel
            for chan_idx in range(n_channels):
                # Mean
                features[:, feature_idx] = np.mean(window_data[:, chan_idx, :], axis=1)
                feature_idx += 1
                
                # Standard deviation
                features[:, feature_idx] = np.std(window_data[:, chan_idx, :], axis=1)
                feature_idx += 1
                
                # Min
                features[:, feature_idx] = np.min(window_data[:, chan_idx, :], axis=1)
                feature_idx += 1
                
                # Max
                features[:, feature_idx] = np.max(window_data[:, chan_idx, :], axis=1)
                feature_idx += 1
        
        # Store in features dictionary
        self.features[f"{participant_id}_time_domain"] = features
        
        return features
    
    def extract_csp_features(self, participant_id, labels, n_components=4, fmin=8, fmax=30):
        """
        Extract Common Spatial Pattern (CSP) features from epoched EEG data.
        This is particularly useful for motor imagery classification.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
        labels : array-like
            Binary class labels for each epoch
        n_components : int
            Number of CSP components to extract
        fmin : float
            Minimum frequency for filtering
        fmax : float
            Maximum frequency for filtering
            
        Returns
        -------
        numpy.ndarray
            Array of shape (n_epochs, n_components) containing CSP features
        """
        if participant_id not in self.eeg.epochs:
            warnings.warn(f"No epochs data found for participant {participant_id}")
            return None
        
        epochs = self.eeg.epochs[participant_id]
        
        # Filter data in the specified frequency band
        filtered_epochs = epochs.copy().filter(fmin, fmax)
        
        # Get data and ensure we have binary labels
        X = filtered_epochs.get_data()
        y = np.array(labels)
        
        # Fit CSP
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        csp.fit(X, y)
        
        # Transform the data
        features = csp.transform(X)
        
        # Store in features dictionary
        self.features[f"{participant_id}_csp"] = features
        
        # Also store the fitted CSP for later use
        self.features[f"{participant_id}_csp_filter"] = csp
        
        return features
    
    def combine_features(self, participant_id, feature_types=None):
        """
        Combine multiple feature types into a single feature matrix.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
        feature_types : list, optional
            List of feature types to combine
            
        Returns
        -------
        numpy.ndarray
            Combined feature matrix
        """
        # Get all available feature types for the participant
        available_features = [key for key in self.features.keys() if key.startswith(participant_id)]
        
        if not available_features:
            warnings.warn(f"No features found for participant {participant_id}")
            return None
        
        # If feature types not specified, use all available
        if feature_types is None:
            feature_types = available_features
        else:
            # Make sure all requested feature types are available
            feature_types = [f"{participant_id}_{ft}" if not ft.startswith(participant_id) else ft 
                            for ft in feature_types]
            for ft in feature_types:
                if ft not in available_features:
                    warnings.warn(f"Feature type {ft} not found for participant {participant_id}")
                    return None
        
        # Combine features
        combined = np.hstack([self.features[ft] for ft in feature_types])
        
        return combined
    
    def create_feature_dataset(self, participants, feature_types=None, labels=None):
        """
        Create a feature dataset for multiple participants.
        
        Parameters
        ----------
        participants : list
            List of participant IDs to include
        feature_types : list, optional
            List of feature types to combine
        labels : dict, optional
            Dictionary mapping participant IDs to label arrays
            
        Returns
        -------
        tuple
            X (features) and y (labels) arrays
        """
        X_list = []
        y_list = []
        
        for participant_id in participants:
            # Get features for this participant
            X_participant = self.combine_features(participant_id, feature_types)
            
            if X_participant is None:
                continue
            
            X_list.append(X_participant)
            
            # Get labels if provided
            if labels is not None and participant_id in labels:
                y_participant = labels[participant_id]
                if len(y_participant) != X_participant.shape[0]:
                    warnings.warn(f"Number of labels ({len(y_participant)}) doesn't match number of feature vectors ({X_participant.shape[0]}) for participant {participant_id}")
                    continue
                y_list.append(y_participant)
        
        if not X_list:
            warnings.warn("No features found for any participant")
            return None, None
        
        # Combine features across participants
        X = np.vstack(X_list)
        
        # Combine labels if provided
        y = None
        if y_list:
            y = np.concatenate(y_list)
        
        return X, y


class MLClassification:
    """
    Class for machine learning classification of EEG data.
    """
    
    def __init__(self):
        """Initialize the classification."""
        self.models = {}
        self.results = {}
    
    def train_test_split(self, X, y, test_size=0.3, random_state=42):
        """
        Split data into training and testing sets.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Labels
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def train_classifier(self, X_train, y_train, model_type='svm', model_params=None, standardize=True, feature_selection=None):
        """
        Train a classifier on the given data.
        
        Parameters
        ----------
        X_train : numpy.ndarray
            Training feature matrix
        y_train : numpy.ndarray
            Training labels
        model_type : str
            Type of model to train ('svm', 'rf', 'lr')
        model_params : dict, optional
            Model parameters
        standardize : bool
            Whether to standardize features
        feature_selection : int or None
            Number of features to select using ANOVA F-value
            
        Returns
        -------
        sklearn.pipeline.Pipeline
            The trained model pipeline
        """
        # Define default model parameters if not specified
        if model_params is None:
            if model_type == 'svm':
                model_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True}
            elif model_type == 'rf':
                model_params = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
            elif model_type == 'lr':
                model_params = {'C': 1.0, 'penalty': 'l2', 'random_state': 42}
        
        # Create pipeline steps
        steps = []
        
        # Add standardization if requested
        if standardize:
            steps.append(('scaler', StandardScaler()))
        
        # Add feature selection if requested
        if feature_selection is not None:
            steps.append(('feature_selection', SelectKBest(f_classif, k=feature_selection)))
        
        # Add the classifier
        if model_type == 'svm':
            steps.append(('classifier', SVC(**model_params)))
        elif model_type == 'rf':
            steps.append(('classifier', RandomForestClassifier(**model_params)))
        elif model_type == 'lr':
            steps.append(('classifier', LogisticRegression(**model_params)))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create and fit the pipeline
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        
        # Store the model
        model_name = f"{model_type}_{'standardized' if standardize else 'raw'}_{'fs' + str(feature_selection) if feature_selection else 'allfeatures'}"
        self.models[model_name] = pipeline
        
        return pipeline
    
    def evaluate_classifier(self, model, X_test, y_test):
        """
        Evaluate a trained classifier on test data.
        
        Parameters
        ----------
        model : sklearn.pipeline.Pipeline
            The trained model
        X_test : numpy.ndarray
            Test feature matrix
        y_test : numpy.ndarray
            Test labels
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        # Predict classes and probabilities
        y_pred = model.predict(X_test)
        
        # Check if the model supports probability prediction
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None
        
        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Calculate ROC and AUC if probabilities are available
        if y_prob is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            results['roc'] = {'fpr': fpr, 'tpr': tpr}
            results['auc'] = auc(fpr, tpr)
        
        return results
    
    def cross_validate(self, X, y, model_type='svm', model_params=None, standardize=True, feature_selection=None, cv=5):
        """
        Perform cross-validation of a classifier.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Labels
        model_type : str
            Type of model to train
        model_params : dict, optional
            Model parameters
        standardize : bool
            Whether to standardize features
        feature_selection : int or None
            Number of features to select
        cv : int
            Number of cross-validation folds
            
        Returns
        -------
        dict
            Cross-validation results
        """
        # Create pipeline steps
        steps = []
        
        # Add standardization if requested
        if standardize:
            steps.append(('scaler', StandardScaler()))
        
        # Add feature selection if requested
        if feature_selection is not None:
            steps.append(('feature_selection', SelectKBest(f_classif, k=feature_selection)))
        
        # Define default model parameters if not specified
        if model_params is None:
            if model_type == 'svm':
                model_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
            elif model_type == 'rf':
                model_params = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
            elif model_type == 'lr':
                model_params = {'C': 1.0, 'penalty': 'l2', 'random_state': 42}
        
        # Add the classifier
        if model_type == 'svm':
            steps.append(('classifier', SVC(**model_params)))
        elif model_type == 'rf':
            steps.append(('classifier', RandomForestClassifier(**model_params)))
        elif model_type == 'lr':
            steps.append(('classifier', LogisticRegression(**model_params)))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create the pipeline
        pipeline = Pipeline(steps)
        
        # Perform cross-validation
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv_obj, scoring='accuracy')
        
        # Store and return results
        model_name = f"{model_type}_cv{cv}_{'standardized' if standardize else 'raw'}_{'fs' + str(feature_selection) if feature_selection else 'allfeatures'}"
        self.results[model_name] = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
        
        return self.results[model_name]
    
    def feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance from a trained model.
        
        Parameters
        ----------
        model_name : str
            Name of the trained model
        feature_names : list, optional
            Names of the features
            
        Returns
        -------
        pandas.DataFrame
            Feature importance scores
        """
        if model_name not in self.models:
            warnings.warn(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        # Get the classifier from the pipeline
        classifier = model.named_steps['classifier']
        
        # Get feature importance based on classifier type
        if isinstance(classifier, RandomForestClassifier):
            importance = classifier.feature_importances_
        elif isinstance(classifier, LogisticRegression):
            importance = np.abs(classifier.coef_[0])
        else:
            warnings.warn(f"Feature importance not available for {type(classifier)}")
            return None
        
        # If feature selection was used, we need to map back to original features
        if 'feature_selection' in model.named_steps:
            # Get the selected feature indices
            selected_indices = model.named_steps['feature_selection'].get_support(indices=True)
            
            # Create a mask of selected features
            mask = np.zeros(feature_names.shape if feature_names is not None else (X.shape[1],), dtype=bool)
            mask[selected_indices] = True
            
            # Create the importance array with zeros for non-selected features
            full_importance = np.zeros(mask.shape)
            full_importance[mask] = importance
            importance = full_importance
        
        # Create feature importance DataFrame
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
        else:
            importance_df = pd.DataFrame({
                'Feature': [f"Feature_{i}" for i in range(len(importance))],
                'Importance': importance
            })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_name, filename):
        """
        Save a trained model to disk.
        
        Parameters
        ----------
        model_name : str
            Name of the trained model
        filename : str
            Filename to save the model to
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if model_name not in self.models:
            warnings.warn(f"Model {model_name} not found")
            return False
        
        try:
            joblib.dump(self.models[model_name], filename)
            print(f"Model saved to {filename}")
            return True
        except Exception as e:
            warnings.warn(f"Error saving model: {e}")
            return False
    
    def load_model(self, filename, model_name=None):
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        filename : str
            Filename to load the model from
        model_name : str, optional
            Name to give the loaded model
            
        Returns
        -------
        sklearn.pipeline.Pipeline
            The loaded model
        """
        try:
            model = joblib.load(filename)
            if model_name is None:
                model_name = os.path.basename(filename).split('.')[0]
            self.models[model_name] = model
            print(f"Model loaded from {filename}")
            return model
        except Exception as e:
            warnings.warn(f"Error loading model: {e}")
            return None


class MLVisualization:
    """
    Class for visualizing machine learning results.
    """
    
    def __init__(self, ml_classification=None):
        """
        Initialize the visualization.
        
        Parameters
        ----------
        ml_classification : MLClassification, optional
            The classification object with models and results
        """
        self.ml = ml_classification
    
    def plot_confusion_matrix(self, confusion_matrix, class_names=None, title="Confusion Matrix", cmap="Blues", normalize=True, save_path=None):
        """
        Plot a confusion matrix.
        
        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            The confusion matrix to plot
        class_names : list, optional
            Names of the classes
        title : str
            Title of the plot
        cmap : str
            Colormap to use
        normalize : bool
            Whether to normalize the confusion matrix
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        plt.figure(figsize=(8, 6))
        
        # Normalize if requested
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, fpr, tpr, auc_value, title="ROC Curve", save_path=None):
        """
        Plot a ROC curve.
        
        Parameters
        ----------
        fpr : numpy.ndarray
            False positive rates
        tpr : numpy.ndarray
            True positive rates
        auc_value : float
            Area under the curve
        title : str
            Title of the plot
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        plt.figure(figsize=(8, 6))
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_feature_importance(self, importance_df, top_n=20, title="Feature Importance", save_path=None):
        """
        Plot feature importance.
        
        Parameters
        ----------
        importance_df : pandas.DataFrame
            DataFrame with feature importance scores
        top_n : int
            Number of top features to display
        title : str
            Title of the plot
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        plt.figure(figsize=(10, 8))
        
        # Sort and take top N features
        importance_df = importance_df.sort_values('Importance', ascending=True).tail(top_n)
        
        # Plot horizontal bar chart
        sns.barplot(
            data=importance_df,
            y='Feature',
            x='Importance',
            palette='viridis'
        )
        
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_cross_validation_results(self, results, title="Cross-Validation Results", save_path=None):
        """
        Plot cross-validation results.
        
        Parameters
        ----------
        results : dict
            Dictionary of cross-validation results
        title : str
            Title of the plot
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        model_names = list(results.keys())
        mean_scores = [results[model]['mean_score'] for model in model_names]
        std_scores = [results[model]['std_score'] for model in model_names]
        
        # Sort by mean score
        sorted_indices = np.argsort(mean_scores)
        model_names = [model_names[i] for i in sorted_indices]
        mean_scores = [mean_scores[i] for i in sorted_indices]
        std_scores = [std_scores[i] for i in sorted_indices]
        
        # Plot
        plt.barh(model_names, mean_scores, xerr=std_scores, capsize=5, color='skyblue')
        
        plt.title(title)
        plt.xlabel('Accuracy')
        plt.ylabel('Model')
        plt.grid(True, axis='x', alpha=0.3)
        plt.xlim(0, 1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_dimensionality_reduction(self, X, y, method='pca', n_components=2, title=None, save_path=None):
        """
        Plot dimensionality reduction of features.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Labels
        method : str
            Dimensionality reduction method ('pca' or 'tsne')
        n_components : int
            Number of components to reduce to
        title : str, optional
            Title of the plot
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        # Standardize features
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X_scaled)
            var_explained = reducer.explained_variance_ratio_
            if title is None:
                title = f"PCA (Explained Variance: {sum(var_explained):.2%})"
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X_scaled)
            if title is None:
                title = "t-SNE"
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Create a DataFrame for plotting
        df = pd.DataFrame(X_reduced, columns=[f"Component {i+1}" for i in range(n_components)])
        df['Label'] = y
        
        # Generate a color map for the classes
        unique_classes = np.unique(y)
        colors = sns.color_palette("viridis", len(unique_classes))
        color_map = {c: colors[i] for i, c in enumerate(unique_classes)}
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        if n_components == 2:
            # 2D scatter plot
            sns.scatterplot(
                data=df,
                x="Component 1",
                y="Component 2",
                hue="Label",
                palette=color_map,
                s=100,
                alpha=0.7
            )
        elif n_components == 3:
            # 3D scatter plot
            ax = plt.figure(figsize=(10, 8)).add_subplot(111, projection='3d')
            for label in unique_classes:
                mask = df['Label'] == label
                ax.scatter(
                    df.loc[mask, "Component 1"],
                    df.loc[mask, "Component 2"],
                    df.loc[mask, "Component 3"],
                    label=label,
                    s=100,
                    alpha=0.7
                )
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            ax.legend()
        else:
            # For higher dimensions, show the first two components
            warnings.warn(f"Plotting only first 2 components of {n_components}")
            sns.scatterplot(
                data=df,
                x="Component 1",
                y="Component 2",
                hue="Label",
                palette=color_map,
                s=100,
                alpha=0.7
            )
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


class NavigationML:
    """
    High-level class for machine learning analysis of spatial navigation EEG data.
    """
    
    def __init__(self, dataset, eeg_preprocessing=None):
        """
        Initialize the navigation ML analysis.
        
        Parameters
        ----------
        dataset : SpatialNavDataset
            The dataset containing behavioral and trigger information
        eeg_preprocessing : EEGPreprocessing, optional
            Preprocessing object with loaded EEG data
        """
        self.dataset = dataset
        self.eeg = eeg_preprocessing
        self.feature_extraction = None
        self.classification = MLClassification()
        self.visualization = MLVisualization(self.classification)
    
    def prepare_eeg_features(self, participant_ids=None, feature_types=None):
        """
        Prepare EEG features for machine learning.
        
        Parameters
        ----------
        participant_ids : list, optional
            List of participant IDs to include
        feature_types : list, optional
            List of feature types to extract
            
        Returns
        -------
        EEGFeatureExtraction
            The feature extraction object
        """
        # If EEG preprocessing not provided, create it
        if self.eeg is None:
            self.eeg = EEGPreprocessing(self.dataset)
        
        # Initialize feature extraction
        self.feature_extraction = EEGFeatureExtraction(self.eeg)
        
        # Default to all participants if not specified
        if participant_ids is None:
            participant_ids = list(self.eeg.epochs.keys())
        
        # Default feature types if not specified
        if feature_types is None:
            feature_types = ['band_power', 'time_domain']
        
        # Extract features for each participant
        for participant_id in participant_ids:
            if participant_id not in self.eeg.epochs:
                warnings.warn(f"No epochs data found for participant {participant_id}, trying to load...")
                # Try to load raw EEG data
                raw = self.eeg.load_raw_eeg(participant_id)
                if raw is None:
                    warnings.warn(f"Could not load raw EEG data for participant {participant_id}")
                    continue
                
                # Extract epochs
                epochs = self.eeg.extract_epochs(participant_id)
                if epochs is None:
                    warnings.warn(f"Could not extract epochs for participant {participant_id}")
                    continue
            
            # Extract features
            for feature_type in feature_types:
                if feature_type == 'band_power':
                    self.feature_extraction.extract_band_power(participant_id)
                elif feature_type == 'connectivity':
                    self.feature_extraction.extract_connectivity(participant_id)
                elif feature_type == 'time_domain':
                    self.feature_extraction.extract_time_domain_features(participant_id)
                # CSP needs labels, so it's handled separately
        
        return self.feature_extraction
    
    def prepare_behavioral_labels(self, label_type='navigation_type', class_mapping=None):
        """
        Prepare behavioral labels for machine learning.
        
        Parameters
        ----------
        label_type : str
            Type of label to extract ('navigation_type', 'difficulty', 'accuracy')
        class_mapping : dict, optional
            Mapping from original labels to class indices
            
        Returns
        -------
        dict
            Dictionary mapping participant IDs to label arrays
        """
        if self.dataset.main_data is None:
            warnings.warn("No behavioral data found")
            return None
        
        # Check if the label type exists in the data
        if label_type not in self.dataset.main_data.columns:
            warnings.warn(f"Label type {label_type} not found in dataset")
            return None
        
        # Prepare labels for each participant
        labels = {}
        
        for participant_id in self.dataset.main_data['participant_id'].unique():
            participant_data = self.dataset.main_data[self.dataset.main_data['participant_id'] == participant_id]
            
            # Extract labels
            y = participant_data[label_type].values
            
            # Apply class mapping if provided
            if class_mapping is not None:
                y = np.array([class_mapping.get(label, label) for label in y])
            
            labels[participant_id] = y
        
        return labels
    
    def classify_navigation_strategies(self, feature_types=None, participant_ids=None, test_size=0.3, model_types=None):
        """
        Classify navigation strategies (egocentric vs. allocentric) using machine learning.
        
        Parameters
        ----------
        feature_types : list, optional
            List of feature types to use
        participant_ids : list, optional
            List of participant IDs to include
        test_size : float
            Proportion of data to use for testing
        model_types : list, optional
            List of model types to try
            
        Returns
        -------
        dict
            Classification results
        """
        # Ensure feature extraction has been performed
        if self.feature_extraction is None:
            warnings.warn("Feature extraction has not been performed")
            return None
        
        # Default to all participants with features if not specified
        if participant_ids is None:
            # Get all participant IDs with features
            feature_keys = list(self.feature_extraction.features.keys())
            participant_ids = list(set([key.split('_')[0] for key in feature_keys]))
        
        # Default model types if not specified
        if model_types is None:
            model_types = ['svm', 'rf', 'lr']
        
        # Prepare labels for navigation type
        # Assuming navigation_type has values like 'egocentric' and 'allocentric'
        # Map them to 0 and 1 for classification
        unique_nav_types = self.dataset.main_data['navigation_type'].unique()
        class_mapping = {nav_type: i for i, nav_type in enumerate(unique_nav_types)}
        
        labels = self.prepare_behavioral_labels('navigation_type', class_mapping)
        if labels is None:
            warnings.warn("Could not prepare labels")
            return None
        
        # Create feature dataset
        X, y = self.feature_extraction.create_feature_dataset(participant_ids, feature_types, labels)
        if X is None or y is None:
            warnings.warn("Could not create feature dataset")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = self.classification.train_test_split(X, y, test_size=test_size)
        
        # Train and evaluate models
        results = {}
        
        for model_type in model_types:
            # Train the model
            model = self.classification.train_classifier(
                X_train,
                y_train,
                model_type=model_type,
                standardize=True,
                feature_selection=min(100, X.shape[1])  # Select top 100 features or all if less
            )
            
            # Evaluate the model
            eval_results = self.classification.evaluate_classifier(model, X_test, y_test)
            
            # Add to results
            results[model_type] = eval_results
            
            # Visualize confusion matrix
            self.visualization.plot_confusion_matrix(
                eval_results['confusion_matrix'],
                class_names=list(unique_nav_types),
                title=f"Confusion Matrix - {model_type.upper()}"
            )
            
            # Visualize ROC curve if available
            if 'roc' in eval_results:
                self.visualization.plot_roc_curve(
                    eval_results['roc']['fpr'],
                    eval_results['roc']['tpr'],
                    eval_results['auc'],
                    title=f"ROC Curve - {model_type.upper()}"
                )
        
        # Visualize feature importance for random forest
        if 'rf' in results:
            rf_model_name = next((name for name in self.classification.models.keys() if name.startswith('rf')), None)
            if rf_model_name:
                # Create feature names
                feature_names = []
                if feature_types is not None:
                    for ft in feature_types:
                        if ft == 'band_power':
                            bands = ['theta', 'alpha', 'beta', 'gamma']
                            channels = self.eeg.epochs[participant_ids[0]].ch_names
                            for band in bands:
                                for ch in channels:
                                    feature_names.append(f"{band}_{ch}")
                        elif ft == 'time_domain':
                            windows = ['early', 'middle', 'late']
                            channels = self.eeg.epochs[participant_ids[0]].ch_names
                            metrics = ['mean', 'std', 'min', 'max']
                            for window in windows:
                                for ch in channels:
                                    for metric in metrics:
                                        feature_names.append(f"{window}_{ch}_{metric}")
                
                if not feature_names:
                    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
                
                # Check if feature names match feature count
                if len(feature_names) != X.shape[1]:
                    warnings.warn(f"Feature names count ({len(feature_names)}) doesn't match feature count ({X.shape[1]})")
                    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
                
                # Get feature importance
                importance_df = self.classification.feature_importance(rf_model_name, np.array(feature_names))
                
                # Visualize feature importance
                self.visualization.plot_feature_importance(
                    importance_df,
                    top_n=20,
                    title="Feature Importance - Random Forest"
                )
        
        # Visualize dimensionality reduction
        self.visualization.plot_dimensionality_reduction(
            X,
            y,
            method='pca',
            title="PCA of EEG Features - Navigation Strategies"
        )
        
        return results
    
    def predict_performance(self, feature_types=None, participant_ids=None, performance_metric='accuracy', test_size=0.3):
        """
        Predict performance (accuracy/RT) based on EEG features.
        
        Parameters
        ----------
        feature_types : list, optional
            List of feature types to use
        participant_ids : list, optional
            List of participant IDs to include
        performance_metric : str
            Performance metric to predict ('accuracy' or 'rt')
        test_size : float
            Proportion of data to use for testing
            
        Returns
        -------
        dict
            Prediction results
        """
        # Ensure feature extraction has been performed
        if self.feature_extraction is None:
            warnings.warn("Feature extraction has not been performed")
            return None
        
        # Default to all participants with features if not specified
        if participant_ids is None:
            # Get all participant IDs with features
            feature_keys = list(self.feature_extraction.features.keys())
            participant_ids = list(set([key.split('_')[0] for key in feature_keys]))
        
        # Convert performance metric to binary classification problem
        # For accuracy: 1 = correct, 0 = incorrect
        # For RT: 1 = fast (below median), 0 = slow (above median)
        
        if performance_metric == 'accuracy':
            # Already binary
            labels = self.prepare_behavioral_labels('accuracy')
        elif performance_metric == 'rt':
            # Convert to binary based on median
            rt_labels = {}
            for participant_id in participant_ids:
                participant_data = self.dataset.main_data[self.dataset.main_data['participant_id'] == participant_id]
                rt = participant_data['rt'].values
                median_rt = np.median(rt)
                rt_labels[participant_id] = (rt < median_rt).astype(int)
            labels = rt_labels
        else:
            warnings.warn(f"Unknown performance metric: {performance_metric}")
            return None
        
        if labels is None:
            warnings.warn("Could not prepare labels")
            return None
        
        # Create feature dataset
        X, y = self.feature_extraction.create_feature_dataset(participant_ids, feature_types, labels)
        if X is None or y is None:
            warnings.warn("Could not create feature dataset")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = self.classification.train_test_split(X, y, test_size=test_size)
        
        # Train and evaluate models
        model_types = ['svm', 'rf', 'lr']
        results = {}
        
        for model_type in model_types:
            # Train the model
            model = self.classification.train_classifier(
                X_train,
                y_train,
                model_type=model_type,
                standardize=True,
                feature_selection=min(100, X.shape[1])
            )
            
            # Evaluate the model
            eval_results = self.classification.evaluate_classifier(model, X_test, y_test)
            
            # Add to results
            results[model_type] = eval_results
            
            # Visualize confusion matrix
            if performance_metric == 'accuracy':
                class_names = ['Incorrect', 'Correct']
            else:  # rt
                class_names = ['Slow', 'Fast']
            
            self.visualization.plot_confusion_matrix(
                eval_results['confusion_matrix'],
                class_names=class_names,
                title=f"Confusion Matrix - {model_type.upper()} - {performance_metric.upper()}"
            )
            
            # Visualize ROC curve if available
            if 'roc' in eval_results:
                self.visualization.plot_roc_curve(
                    eval_results['roc']['fpr'],
                    eval_results['roc']['tpr'],
                    eval_results['auc'],
                    title=f"ROC Curve - {model_type.upper()} - {performance_metric.upper()}"
                )
        
        return results
    
    def cross_subject_classification(self, feature_types=None, label_type='navigation_type', cv=5):
        """
        Perform cross-subject classification.
        
        Parameters
        ----------
        feature_types : list, optional
            List of feature types to use
        label_type : str
            Type of label to classify
        cv : int
            Number of cross-validation folds
            
        Returns
        -------
        dict
            Cross-validation results
        """
        # Ensure feature extraction has been performed
        if self.feature_extraction is None:
            warnings.warn("Feature extraction has not been performed")
            return None
        
        # Get all participant IDs with features
        feature_keys = list(self.feature_extraction.features.keys())
        participant_ids = list(set([key.split('_')[0] for key in feature_keys]))
        
        # Prepare labels
        unique_values = self.dataset.main_data[label_type].unique()
        class_mapping = {val: i for i, val in enumerate(unique_values)}
        
        labels = self.prepare_behavioral_labels(label_type, class_mapping)
        if labels is None:
            warnings.warn("Could not prepare labels")
            return None
        
        # Results dictionary
        results = {}
        
        # Leave-one-subject-out cross-validation
        for test_subject in participant_ids:
            # Training subjects
            train_subjects = [subj for subj in participant_ids if subj != test_subject]
            
            # Create feature datasets
            X_train, y_train = self.feature_extraction.create_feature_dataset(train_subjects, feature_types, labels)
            X_test, y_test = self.feature_extraction.create_feature_dataset([test_subject], feature_types, labels)
            
            if X_train is None or y_train is None or X_test is None or y_test is None:
                warnings.warn(f"Could not create feature dataset for subject {test_subject}")
                continue
            
            # Train classifier
            model = self.classification.train_classifier(
                X_train,
                y_train,
                model_type='svm',  # Use SVM for cross-subject
                standardize=True,
                feature_selection=min(100, X_train.shape[1])
            )
            
            # Evaluate on test subject
            eval_results = self.classification.evaluate_classifier(model, X_test, y_test)
            
            # Store results
            results[test_subject] = eval_results['accuracy']
        
        # Calculate average cross-subject accuracy
        avg_accuracy = np.mean(list(results.values()))
        std_accuracy = np.std(list(results.values()))
        
        # Print results
        print(f"Cross-subject classification results for {label_type}:")
        print(f"Average accuracy: {avg_accuracy:.2f} ± {std_accuracy:.2f}")
        for subject, acc in results.items():
            print(f"Subject {subject}: {acc:.2f}")
        
        return {
            'subject_accuracies': results,
            'mean_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy
        }
    
    def identify_neural_signatures(self, feature_types=None, participant_ids=None, navigation_types=None):
        """
        Identify neural signatures of different navigation strategies.
        
        Parameters
        ----------
        feature_types : list, optional
            List of feature types to use
        participant_ids : list, optional
            List of participant IDs to include
        navigation_types : list, optional
            List of navigation types to compare
            
        Returns
        -------
        dict
            Neural signature results
        """
        # Ensure feature extraction has been performed
        if self.feature_extraction is None:
            warnings.warn("Feature extraction has not been performed")
            return None
        
        # Default to all participants with features if not specified
        if participant_ids is None:
            # Get all participant IDs with features
            feature_keys = list(self.feature_extraction.features.keys())
            participant_ids = list(set([key.split('_')[0] for key in feature_keys]))
        
        # Default navigation types if not specified
        if navigation_types is None:
            navigation_types = self.dataset.main_data['navigation_type'].unique()
        
        # Prepare labels
        labels = self.prepare_behavioral_labels('navigation_type')
        if labels is None:
            warnings.warn("Could not prepare labels")
            return None
        
        # Results dictionary
        signatures = {}
        
        # For each navigation type
        for nav_type in navigation_types:
            # Create binary labels: 1 for the current navigation type, 0 for others
            binary_labels = {}
            for subj in participant_ids:
                if subj in labels:
                    binary_labels[subj] = (labels[subj] == nav_type).astype(int)
            
            # Create feature dataset
            X, y = self.feature_extraction.create_feature_dataset(participant_ids, feature_types, binary_labels)
            if X is None or y is None:
                warnings.warn(f"Could not create feature dataset for navigation type {nav_type}")
                continue
            
            # Find discriminative features
            # Train a random forest to identify important features
            model = self.classification.train_classifier(
                X,
                y,
                model_type='rf',
                standardize=True
            )
            
            # Get feature importance
            rf_model_name = next((name for name in self.classification.models.keys() if name.startswith('rf')), None)
            if rf_model_name:
                # Create feature names
                feature_names = []
                if feature_types is not None:
                    for ft in feature_types:
                        if ft == 'band_power':
                            bands = ['theta', 'alpha', 'beta', 'gamma']
                            channels = self.eeg.epochs[participant_ids[0]].ch_names
                            for band in bands:
                                for ch in channels:
                                    feature_names.append(f"{band}_{ch}")
                        elif ft == 'time_domain':
                            windows = ['early', 'middle', 'late']
                            channels = self.eeg.epochs[participant_ids[0]].ch_names
                            metrics = ['mean', 'std', 'min', 'max']
                            for window in windows:
                                for ch in channels:
                                    for metric in metrics:
                                        feature_names.append(f"{window}_{ch}_{metric}")
                
                if not feature_names:
                    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
                
                # Check if feature names match feature count
                if len(feature_names) != X.shape[1]:
                    warnings.warn(f"Feature names count ({len(feature_names)}) doesn't match feature count ({X.shape[1]})")
                    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
                
                # Get feature importance
                importance_df = self.classification.feature_importance(rf_model_name, np.array(feature_names))
                
                # Store top features as neural signatures
                signatures[nav_type] = importance_df.head(20)
                
                # Visualize feature importance
                self.visualization.plot_feature_importance(
                    importance_df,
                    top_n=20,
                    title=f"Neural Signature - {nav_type}"
                )
        
        return signatures
    
    def compare_strategies_over_time(self, feature_types=None, participant_ids=None, time_windows=None):
        """
        Compare neural signatures of navigation strategies over time.
        
        Parameters
        ----------
        feature_types : list, optional
            List of feature types to use
        participant_ids : list, optional
            List of participant IDs to include
        time_windows : list, optional
            List of time windows to compare
            
        Returns
        -------
        dict
            Time-based comparison results
        """
        # Ensure feature extraction has been performed
        if self.feature_extraction is None:
            warnings.warn("Feature extraction has not been performed")
            return None
        
        # Default to all participants with features if not specified
        if participant_ids is None:
            # Get all participant IDs with features
            feature_keys = list(self.feature_extraction.features.keys())
            participant_ids = list(set([key.split('_')[0] for key in feature_keys]))
        
        # Default time windows if not specified
        if time_windows is None:
            time_windows = [(0, 0.2), (0.2, 0.4), (0.4, 0.6)]
        
        # Prepare labels
        navigation_types = self.dataset.main_data['navigation_type'].unique()
        class_mapping = {nav_type: i for i, nav_type in enumerate(navigation_types)}
        
        labels = self.prepare_behavioral_labels('navigation_type', class_mapping)
        if labels is None:
            warnings.warn("Could not prepare labels")
            return None
        
        # Results dictionary
        results = {}
        
        # For each time window
        for i, (tmin, tmax) in enumerate(time_windows):
            # Extract time-specific features
            for participant_id in participant_ids:
                if participant_id not in self.eeg.epochs:
                    continue
                
                # Extract time domain features for this specific window
                self.feature_extraction.extract_time_domain_features(
                    participant_id,
                    time_windows=[(tmin, tmax)]
                )
            
            # Create feature dataset for this time window
            X, y = self.feature_extraction.create_feature_dataset(
                participant_ids,
                [f"{participant_id}_time_domain" for participant_id in participant_ids],
                labels
            )
            
            if X is None or y is None:
                warnings.warn(f"Could not create feature dataset for time window {i+1}")
                continue
            
            # Train classifier for this time window
            model = self.classification.train_classifier(
                X,
                y,
                model_type='svm',
                standardize=True,
                feature_selection=min(50, X.shape[1])
            )
            
            # Cross-validate
            cv_results = self.classification.cross_validate(
                X,
                y,
                model_type='svm',
                standardize=True,
                feature_selection=min(50, X.shape[1]),
                cv=5
            )
            
            # Store results
            results[f"window_{i+1}"] = {
                'time_range': (tmin, tmax),
                'accuracy': cv_results['mean_score'],
                'std': cv_results['std_score']
            }
        
        # Visualize time window comparison
        plt.figure(figsize=(10, 6))
        
        window_labels = [f"{tmin}-{tmax}s" for tmin, tmax in time_windows]
        accuracies = [results[f"window_{i+1}"]['accuracy'] for i in range(len(time_windows))]
        stds = [results[f"window_{i+1}"]['std'] for i in range(len(time_windows))]
        
        plt.bar(window_labels, accuracies, yerr=stds, capsize=10, color='skyblue')
        plt.ylim(0, 1)
        plt.xlabel('Time Window')
        plt.ylabel('Classification Accuracy')
        plt.title('Navigation Strategy Classification by Time Window')
        plt.grid(True, axis='y', alpha=0.3)
        
        return results


def run_ml_analysis(data_dir, output_dir=None):
    """
    Run machine learning analysis on spatial navigation EEG data.
    
    Parameters
    ----------
    data_dir : str
        Path to the directory containing all data files
    output_dir : str, optional
        Path to the directory to save output files
    """
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    dataset = SpatialNavDataset(data_dir)
    dataset.summarize()
    
    # Initialize EEG preprocessing
    eeg = EEGPreprocessing(dataset)
    
    # Load EEG data for each participant
    participant_ids = dataset.main_data['participant_id'].unique()
    
    print("\n=== Loading EEG Data ===")
    for participant_id in participant_ids[:2]:  # Limit to first 2 participants for example
        print(f"Loading data for participant {participant_id}...")
        eeg.load_raw_eeg(participant_id)
        eeg.extract_epochs(participant_id)
    
    # Initialize Navigation ML
    nav_ml = NavigationML(dataset, eeg)
    
    print("\n=== Extracting EEG Features ===")
    # Extract features
    nav_ml.prepare_eeg_features(participant_ids=participant_ids[:2], feature_types=['band_power', 'time_domain'])
    
    print("\n=== Classifying Navigation Strategies ===")
    # Classify navigation strategies
    strategy_results = nav_ml.classify_navigation_strategies()
    
    if strategy_results:
        print("\nNavigation Strategy Classification Results:")
        for model_type, results in strategy_results.items():
            print(f"{model_type.upper()}: Accuracy = {results['accuracy']:.2f}")
    
    print("\n=== Predicting Performance ===")
    # Predict performance
    performance_results = nav_ml.predict_performance(performance_metric='accuracy')
    
    if performance_results:
        print("\nPerformance Prediction Results (Accuracy):")
        for model_type, results in performance_results.items():
            print(f"{model_type.upper()}: Accuracy = {results['accuracy']:.2f}")
    
    print("\n=== Identifying Neural Signatures ===")
    # Identify neural signatures
    signatures = nav_ml.identify_neural_signatures()
    
    if signatures:
        print("\nNeural Signatures:")
        for nav_type, sig_df in signatures.items():
            print(f"\nTop 5 features for {nav_type}:")
            print(sig_df.head(5))
    
    print("\n=== Comparing Strategies Over Time ===")
    # Compare strategies over time
    time_results = nav_ml.compare_strategies_over_time()
    
    if time_results:
        print("\nTime Window Comparison Results:")
        for window, results in time_results.items():
            print(f"{window} ({results['time_range'][0]}-{results['time_range'][1]}s): Accuracy = {results['accuracy']:.2f} ± {results['std']:.2f}")
    
    print("\n=== ML Analysis Complete ===")
    if output_dir is not None:
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/your/data"
    output_dir = "path/to/your/output"
    
    run_ml_analysis(data_dir, output_dir)