import sys,tempfile,unittest
from pathlib import Path
import torch
from rsl_rl.algorithms import PPO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from joint_strike_core import JointStrikeActor
from strike_actor_restore import restore_actor

class ActorRestoreContract(unittest.TestCase):
    def test_installed_ppo_api_and_fresh_critic(self):
        old=JointStrikeActor(8,8,30);new=JointStrikeActor(8,8,30)
        with torch.no_grad():old.actor[-1].bias.fill_(.123)
        before={k:v.clone() for k,v in new.state_dict().items() if k.startswith('critic.')}
        algorithm=PPO(new,device='cpu')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt';torch.save({'model_state_dict':old.state_dict()},p)
            restore_actor(algorithm,p,'cpu')
        torch.testing.assert_close(new.actor[-1].bias,old.actor[-1].bias)
        for key,value in before.items():self.assertTrue(torch.equal(new.state_dict()[key],value))
        self.assertFalse(algorithm.optimizer.state)

if __name__=='__main__':unittest.main()
