import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.threshold = 0.0
        self.X_test_processed = None
        self.y_test = None
        
        self.columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
                        "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
                        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
                        "num_shells","num_access_files","num_outbound_cmds","is_host_login",
                        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
                        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
                        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

    def load_data(self, file_path):
        df = pd.read_csv(file_path, names=self.columns, low_memory=False)
        # Fix string-to-float errors by removing header repetitions
        df = df[df['duration'] != 'duration']
        
        # Force numeric conversion
        numeric_cols = df.columns.drop(['protocol_type', 'service', 'flag', 'label'])
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Categorical Encoding
        le = LabelEncoder()
        for col in ['protocol_type', 'service', 'flag']:
            df[col] = le.fit_transform(df[col].astype(str))
            
        df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
        return df

    def preprocess(self, df):
        labels = df['label']
        data = df.drop(['label'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        
        self.X_train_processed = self.scaler.fit_transform(X_train)
        self.X_test_processed = self.scaler.transform(X_test)
        self.y_test = y_test
        self.X_train_normal = self.X_train_processed[y_train.values == 0]

    def build_and_train(self, epochs=50):
        input_dim = self.X_train_normal.shape[1]
        inp = Input(shape=(input_dim,))
        enc = Dense(16, activation='relu')(inp)
        dec = Dense(input_dim, activation='sigmoid')(enc)
        
        self.model = Model(inp, dec)
        self.model.compile(optimizer='adam', loss='mse')
        history = self.model.fit(
            self.X_train_normal, 
            self.X_train_normal, 
            epochs=epochs, 
            batch_size=32,
            validation_split=0.1,  # <--- NEW: Keeps 10% data secret for testing
            verbose=0
        )  
        recons = self.model.predict(self.X_train_normal)
        mse = np.mean(np.power(self.X_train_normal - recons, 2), axis=1)
        self.threshold = np.percentile(mse, 95)
        return history.history

    def detect(self):
        recons = self.model.predict(self.X_test_processed)
        mse = np.mean(np.power(self.X_test_processed - recons, 2), axis=1)
        preds = (mse > self.threshold).astype(int)
        return preds, mse

    def get_summary_counts(self, predictions):
        return {"total": len(predictions), "normal": np.sum(predictions==0), "anomalies": np.sum(predictions==1)}

    def get_metrics(self, predictions):
        return {
            'accuracy': accuracy_score(self.y_test, predictions),
            'precision': precision_score(self.y_test, predictions, zero_division=0),
            'recall': recall_score(self.y_test, predictions, zero_division=0),
            'cm': confusion_matrix(self.y_test, predictions)
        }

    def get_anomaly_reason(self, index):
        sample = self.X_test_processed[index].reshape(1, -1)
        reconstruction = self.model.predict(sample, verbose=0)
        errors = np.abs(sample - reconstruction)[0]
        
        # Map feature index to name
        feature_idx = np.argmax(errors)
        feature_name = self.columns[feature_idx]
        return f"Deviation in {feature_name}"