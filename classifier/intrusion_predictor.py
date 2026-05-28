
import logging
import warnings
import numpy as np
import joblib
from pathlib import Path
log = logging.getLogger(__name__)
IDS_CLASSES = ['Bots', 'Brute Force', 'DDoS', 'DoS', 'Normal Traffic', 'Port Scanning', 'Web Attacks']
BENIGN_CLS = 'Normal Traffic'
BENIGN_IDX = IDS_CLASSES.index(BENIGN_CLS)
CLS_FAMILY = {'Normal Traffic': 'BENIGN', 'Bots': 'Bot', 'Brute Force': 'Brute', 'DDoS': 'DDoS', 'DoS': 'DDoS', 'Port Scanning': 'PortScan', 'Web Attacks': 'Bot'}
FEATURE_NAMES_52 = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Length of Fwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min']
assert len(FEATURE_NAMES_52) == 52, 'Feature list must have exactly 52 entries'

class IntrusionPredictor:

    def __init__(self, model_dir: str='model'):
        base = Path(model_dir)
        log.info(f"Loading IDS models from '{model_dir}/' ...")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.selector = joblib.load(base / 'feature_selector.pkl')
            self.scaler = joblib.load(base / 'scaler.pkl')
            self.lgb = joblib.load(base / 'lightgbm.pkl')
            self.lr = joblib.load(base / 'logistic.pkl')
            self.le = joblib.load(base / 'label_encoder.pkl')
        loaded_cls = list(self.le.classes_)
        assert loaded_cls == IDS_CLASSES, f'Label encoder classes mismatch!\n  Expected: {IDS_CLASSES}\n  Got     : {loaded_cls}'
        self._last_lgb_class = BENIGN_CLS
        self._last_lr_class = BENIGN_CLS
        self._last_agree = True
        log.info(f'[IDS] Ready | 52 -> 30 features | 7 classes: {IDS_CLASSES} | LR-primary ensemble')

    def _build_vector(self, metrics: dict) -> np.ndarray:
        raw = []
        for feat in FEATURE_NAMES_52:
            v = metrics.get(feat, 0.0)
            try:
                v = float(v)
            except (ValueError, TypeError):
                v = 0.0
            if v != v or v == float('inf') or v == float('-inf'):
                v = 0.0
            raw.append(v)
        vec = np.array(raw, dtype=np.float64).reshape(1, -1)
        nonzero = int((vec != 0).sum())
        if nonzero < 3:
            log.debug(f'[IDS] Feature vector sparse: {nonzero}/52 non-zero — many metrics may be missing from the API payload.')
        return vec

    def predict_raw(self, metrics: dict):
        vec = self._build_vector(metrics)
        vec_sel = self.selector.transform(vec)
        vec_sel = np.nan_to_num(vec_sel, nan=0.0, posinf=0.0, neginf=0.0)
        vec_sc = self.scaler.transform(vec_sel)
        vec_sc = np.nan_to_num(vec_sc, nan=0.0, posinf=0.0, neginf=0.0)
        lgb_proba = self.lgb.predict_proba(vec_sc)[0].astype(np.float64)
        lr_proba = self.lr.predict_proba(vec_sc)[0].astype(np.float64)
        lgb_idx = int(np.argmax(lgb_proba))
        lr_idx = int(np.argmax(lr_proba))
        both_agree = lgb_idx == lr_idx
        lgb_class = IDS_CLASSES[lgb_idx]
        lr_class = IDS_CLASSES[lr_idx]
        if both_agree:
            ensemble = 0.5 * lr_proba + 0.5 * lgb_proba
            ensemble /= ensemble.sum() + 1e-09
        elif lr_idx != BENIGN_IDX and lgb_idx == BENIGN_IDX:
            ensemble = lr_proba.copy()
        elif lgb_idx != BENIGN_IDX and lr_idx == BENIGN_IDX:
            lr_attack_mass = 1.0 - float(lr_proba[BENIGN_IDX])
            if lr_attack_mass > 0.2:
                ensemble = 0.6 * lr_proba + 0.4 * lgb_proba
                ensemble /= ensemble.sum() + 1e-09
            else:
                ensemble = lr_proba.copy()
        else:
            ensemble = 0.8 * lr_proba + 0.2 * lgb_proba
            ensemble /= ensemble.sum() + 1e-09
        ens_idx = int(np.argmax(ensemble))
        ids_class = IDS_CLASSES[ens_idx]
        rl_family = CLS_FAMILY.get(ids_class, 'BENIGN')
        confidence = float(ensemble[ens_idx])
        if ens_idx == BENIGN_IDX:
            threat_score = 0.0
        else:
            boost = 1.2 if both_agree else 1.0
            threat_score = min(confidence * boost, 1.0)
        self._last_lgb_class = lgb_class
        self._last_lr_class = lr_class
        self._last_agree = both_agree
        log.info(f'[IDS] LR={lr_class}({lr_proba[lr_idx]:.3f}) LGB={lgb_class}({lgb_proba[lgb_idx]:.3f}) -> {ids_class}({confidence:.3f}) agree={both_agree} threat={threat_score:.3f}')
        return (ids_class, rl_family, confidence, threat_score, ensemble, both_agree)

    def predict(self, metrics: dict) -> tuple:
        try:
            ids_class, rl_family, confidence, threat_score, _, both_agree = self.predict_raw(metrics)
            return (rl_family, confidence, threat_score, both_agree)
        except Exception as e:
            log.error(f'IntrusionPredictor.predict() error: {e}', exc_info=True)
            return ('BENIGN', 1.0, 0.0, True)

    def build_rl_probs(self, metrics: dict) -> np.ndarray:
        try:
            ids_class, rl_family, confidence, threat_score, probs_7, both_agree = self.predict_raw(metrics)
            rl_probs = np.zeros(5, dtype=np.float32)
            _IDS_TO_RL = {'Normal Traffic': 0, 'Bots': 1, 'Web Attacks': 1, 'DDoS': 2, 'DoS': 2, 'Port Scanning': 3, 'Brute Force': 4}
            for i, cls in enumerate(IDS_CLASSES):
                rl_idx = _IDS_TO_RL.get(cls, 0)
                rl_probs[rl_idx] += float(probs_7[i])
            rl_probs /= rl_probs.sum() + 1e-09
            return rl_probs
        except Exception as e:
            log.error(f'build_rl_probs error: {e}', exc_info=True)
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def reward_signal(self, metrics: dict) -> dict:
        try:
            ids_class, rl_family, confidence, threat_score, probs_7, both_agree = self.predict_raw(metrics)
            cl = rl_family.lower()
            return {'ids_class': ids_class, 'rl_family': rl_family, 'lgb_class': self._last_lgb_class, 'lr_class': self._last_lr_class, 'models_agree': self._last_agree, 'class_name': rl_family, 'confidence': confidence, 'threat_score': threat_score, 'both_agree': both_agree, 'is_benign': rl_family == 'BENIGN', 'is_ddos': any((k in cl for k in ['ddos', 'dos'])), 'is_brute': any((k in cl for k in ['brute', 'bot'])), 'is_exploit': any((k in cl for k in ['bot', 'web'])), 'is_scan': any((k in cl for k in ['scan', 'portscan'])), 'certainty': confidence * (1.2 if both_agree else 0.9)}
        except Exception as e:
            log.error(f'reward_signal error: {e}', exc_info=True)
            return {'ids_class': 'Normal Traffic', 'rl_family': 'BENIGN', 'lgb_class': 'Normal Traffic', 'lr_class': 'Normal Traffic', 'models_agree': True, 'class_name': 'BENIGN', 'confidence': 1.0, 'threat_score': 0.0, 'both_agree': True, 'is_benign': True, 'is_ddos': False, 'is_brute': False, 'is_exploit': False, 'is_scan': False, 'certainty': 1.0}

    def _build_feature_vec(self, metrics: dict) -> np.ndarray:
        return self._build_vector(metrics)

    @property
    def feature_names(self) -> list:
        return FEATURE_NAMES_52

    @property
    def n_features(self) -> int:
        return 52
