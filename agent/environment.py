import logging
import numpy as np
import gymnasium as gym
from monitor.state_builder import StateBuilder
logger = logging.getLogger(__name__)
ACTION_NAMES = {0: 'Allow', 1: 'Block IP', 2: 'Rate-limit', 3: 'Restart service', 4: 'Flush iptables'}
CLASS_NAMES = ['BENIGN', 'Bot', 'DDoS', 'PortScan', 'Brute']

def compute_reward(probs: np.ndarray, action: int) -> float:
    p_safe = float(probs[0])
    p_bot = float(probs[1])
    p_ddos = float(probs[2])
    p_scan = float(probs[3])
    p_brute = float(probs[4]) if len(probs) > 4 else 0.0
    p_attack = 1.0 - p_safe
    if action == 0:
        reward = p_safe * 1.0 - p_attack * 5.0
    elif action == 1:
        reward = (p_scan + p_bot + p_brute) * 5.0 + p_ddos * 2.0 - p_safe * 3.0
    elif action == 2:
        reward = p_ddos * 6.0 + p_bot * 2.0 + p_brute * 2.0 + p_scan * 1.0 - p_safe * 3.0
    elif action == 3:
        reward = (p_bot + p_ddos + p_brute) * 4.0 - p_safe * 2.0 - p_scan * 1.0
    elif action == 4:
        reward = p_attack * 3.0 - p_safe * 4.0
    else:
        reward = 0.0
    return round(float(reward), 4)

class NetworkDefenseEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config: dict, simulator=None):
        super().__init__()
        self.config = config
        self.simulator = simulator
        self.sb = StateBuilder(config, simulator=simulator)
        self.max_steps = config['training']['max_steps_per_episode']
        self.sim_mode = simulator is not None
        if not self.sim_mode:
            from defense.ubuntu_firewall import UbuntuFirewallDefender
            self.fw = UbuntuFirewallDefender(config)
            logger.info('Live mode: Ubuntu iptables over SSH ✓')
        else:
            self.fw = None
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(config['training']['obs_dim'],), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(config['training']['n_actions'])
        self.step_count = 0
        self.episode_reward = 0.0
        self._last_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fw:
            try:
                self.fw.cleanup_all_rules()
            except Exception as e:
                logger.warning(f'reset cleanup: {e}')
        self.step_count = 0
        self.episode_reward = 0.0
        self._last_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs = self.sb.build_observation()
        return (obs, {})

    def step(self, action: int):
        metrics = self.sb.fetch_metrics()
        fw_result = self._execute_action(action, metrics)
        if self.simulator and action in [1, 2, 3, 4]:
            self.simulator.mark_defended()
        obs = self.sb.build_observation(metrics)
        reward = self._compute_reward(action)
        self.episode_reward += reward
        self.step_count += 1
        done = self.step_count >= self.max_steps
        if not self.sim_mode:
            import time
            time.sleep(1.5)
        class_name = _decode_class(obs)
        if self.sim_mode and fw_result == 'simulated':
            fw_result = f'{ACTION_NAMES.get(action, str(action))} ({class_name})'
        blocked_ips = self.fw.get_blocked() if self.fw else []
        info = {'action_name': ACTION_NAMES[action], 'threat_class': class_name, 'confidence': float(obs[6]), 'threat_score': float(obs[5]), 'blocked_ips': blocked_ips, 'fw_result': fw_result}
        logger.info(f'Step {self.step_count:03d} | action={ACTION_NAMES[action]:<18} | class={class_name:<10} | threat={float(obs[5]):.2f} | conf={float(obs[6]):.2f} | reward={reward:+.3f}')
        return (obs, reward, done, False, info)

    def _execute_action(self, action: int, metrics: dict) -> str:
        if self.sim_mode:
            return 'simulated'
        ip = metrics.get('top_attacker', '')
        subnet = f'{ip}/32' if ip else ''
        try:
            if action == 1 and ip:
                ok = self.fw.block_ip(ip)
                return f'Blocked {ip}' if ok else f'Block failed: {ip}'
            elif action == 2 and subnet:
                ok = self.fw.rate_limit_subnet(subnet)
                return f'Rate-limited {subnet}' if ok else 'Rate-limit failed'
            elif action == 3:
                ok = self.fw.restart_networking()
                return 'Networking restarted' if ok else 'Restart failed'
            elif action == 4:
                self.fw.cleanup_all_rules()
                return 'Flushed iptables'
        except Exception as e:
            logger.warning(f'_execute_action error: {e}')
            return f'error: {e}'
        return 'no-op'

    def _compute_reward(self, action: int) -> float:
        probs = self.sb.get_last_probs()
        reward = compute_reward(probs, action)
        logger.debug(f'Reward: probs={np.round(probs, 3)} action={ACTION_NAMES[action]} reward={reward:+.4f}')
        return reward

def _decode_class(obs: np.ndarray) -> str:
    if obs[15] > 0.5:
        return 'DDoS'
    elif obs[16] > 0.5:
        return 'PortScan'
    elif obs[14] > 0.5:
        return 'Bot'
    elif obs[13] > 0.5:
        return 'BENIGN'
    idx = int(round(float(obs[7]) * (len(CLASS_NAMES) - 1)))
    idx = max(0, min(idx, len(CLASS_NAMES) - 1))
    return CLASS_NAMES[idx]