#import
import logging
import time
import numpy as np
import requests
import warnings
from classifier.intrusion_predictor import IntrusionPredictor
log = logging.getLogger(__name__)
RL_CLASS_NAMES = ['BENIGN', 'Bot', 'DDoS', 'PortScan', 'Brute']

class StateBuilder:

    def __init__(self, config: dict, simulator=None):
        self.simulator = simulator
        self.obs_dim = config['training']['obs_dim']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.ids = IntrusionPredictor(model_dir='model')
        self.feature_names = self.ids.feature_names
        self.n_features = self.ids.n_features
        self._last_metrics = {}
        self._last_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._last_reward_signal = {}
        self._consecutive_failures = 0
        self._session = requests.Session()
        self._session.headers.update({'Connection': 'keep-alive', 'Accept': 'application/json'})
        if simulator is None:
            ip = config['network']['ubuntu_ip']
            port = config['ubuntu']['metrics_port']
            self.url = f'http://{ip}:{port}/state'
        else:
            self.url = None

    def fetch_metrics(self) -> dict:
        if self.simulator is not None:
            m = self.simulator.generate()
            self._last_metrics = m
            return m
        for attempt in range(5):
            try:
                timeout = 15 + attempt * 10
                r = self._session.get(self.url, timeout=timeout)
                r.raise_for_status()
                m = r.json()
                if not isinstance(m, dict) or len(m) < 5:
                    log.warning('API returned unexpected data')
                    continue
                self._last_metrics = m
                self._consecutive_failures = 0
                return m
            except Exception as e:
                self._consecutive_failures += 1
                if attempt < 4:
                    wait = 2 + attempt * 2
                    log.warning(f'fetch_metrics attempt {attempt + 1} failed: {e} (retry in {wait}s)')
                    time.sleep(wait)
                else:
                    log.error(f'fetch_metrics FAILED after 5 attempts: {e}')
        if self._consecutive_failures >= 10:
            log.warning('Recreating HTTP session after 10 consecutive failures')
            self._session.close()
            self._session = requests.Session()
            self._session.headers.update({'Connection': 'keep-alive', 'Accept': 'application/json'})
            self._consecutive_failures = 0
        if self._last_metrics:
            log.warning('Using cached metrics from last successful fetch')
            return dict(self._last_metrics)
        return {}

    def warmup(self, max_wait: int=60) -> bool:
        log.info(f'Warming up API connection ({self.url}) ...')
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                m = self.fetch_metrics()
                if not m:
                    time.sleep(3)
                    continue
                nonzero = sum((1 for f in self.feature_names if float(m.get(f, 0.0)) != 0.0))
                cpu = m.get('cpu_percent', 0)
                mem = m.get('mem_percent', 0)
                if nonzero >= 3 and (cpu > 0 or mem > 0):
                    log.info(f'API warm: {nonzero}/{self.n_features} features non-zero, cpu={cpu:.1f}% mem={mem:.1f}%')
                    return True
                log.info(f'API not ready: {nonzero} non-zero — waiting...')
                time.sleep(3)
            except Exception as e:
                log.warning(f'Warmup attempt failed: {e}')
                time.sleep(3)
        log.error(f'API warmup timed out after {max_wait}s')
        return False

    def build_observation(self, metrics: dict=None) -> np.ndarray:
        if not metrics:
            metrics = self.fetch_metrics()
        rl_class, confidence, threat_score, rl_probs, models_agree = self._classify(metrics)
        self._last_probs = rl_probs
        class_id = RL_CLASS_NAMES.index(rl_class) if rl_class in RL_CLASS_NAMES else 0
        obs = np.array([float(rl_probs[0]), float(rl_probs[1]), float(rl_probs[2]), float(rl_probs[3]), float(rl_probs[4]), threat_score, confidence, class_id / max(len(RL_CLASS_NAMES) - 1, 1), 1.0 if models_agree else 0.0, metrics.get('cpu_percent', 0.0) / 100.0, metrics.get('mem_percent', 0.0) / 100.0, min(metrics.get('bytes_recv', 0) / 100000000.0, 1.0), min(metrics.get('bytes_sent', 0) / 100000000.0, 1.0), 1.0 if rl_class == 'BENIGN' else 0.0, 1.0 if rl_class == 'Bot' else 0.0, 1.0 if rl_class == 'DDoS' else 0.0, 1.0 if rl_class == 'PortScan' else 0.0], dtype=np.float32)
        log.info(f"[OBS] ml_class={rl_class} conf={confidence:.3f} threat={threat_score:.3f} | probs=[B={rl_probs[0]:.2f} Bot={rl_probs[1]:.2f} DDoS={rl_probs[2]:.2f} Scan={rl_probs[3]:.2f} Brute={rl_probs[4]:.2f}] | cpu={metrics.get('cpu_percent', 0):.0f}%")
        return obs

    def _classify(self, metrics: dict):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ids_class, rl_family, confidence, threat_score, probs_7, both_agree = self.ids.predict_raw(metrics)
                rl_probs = self.ids.build_rl_probs(metrics)
            self._last_reward_signal = {'ids_class': ids_class, 'rl_family': rl_family, 'lgb_class': self.ids._last_lgb_class, 'lr_class': self.ids._last_lr_class, 'models_agree': self.ids._last_agree, 'confidence': confidence, 'threat_score': threat_score}
            self._last_metrics.update({'_ml_class': rl_family, '_ml_confidence': confidence, '_ml_threat': threat_score})
            log.info(f"[ML] IDS: LR={self._last_reward_signal.get('lr_class', '?')} LGB={self._last_reward_signal.get('lgb_class', '?')} -> {ids_class}({confidence:.3f}) RL={rl_family} threat={threat_score:.3f} agree={both_agree}")
            return (rl_family, confidence, threat_score, rl_probs, both_agree)
        except Exception as e:
            log.error(f'_classify error: {e}', exc_info=True)
            return ('BENIGN', 0.5, 0.0, np.array([1, 0, 0, 0, 0], dtype=np.float32), True)

    def get_last_probs(self) -> np.ndarray:
        return self._last_probs.copy()

    def get_top_attacker_ip(self) -> str:
        if self.simulator is not None:
            return getattr(self.simulator, '_top_ip', '')
        return self._last_metrics.get('top_attacker', '')

    def get_last_reward_signal(self) -> dict:
        return dict(self._last_reward_signal)
