from dataclasses import dataclass
from typing import (Any, Dict, List, Optional, Tuple)
import numpy as np
import torch
from torch import optim, nn
import pickle
from itertools import product
from tqdm import tqdm

from algorithm.encoder import Encoder, normalize_embeds
from algorithm.reporter import get_reporter
from algorithm.envs_tasks_desc import ENV_UUID, SKILL_UUID, TASK_UUID, gen_uuid
from algorithm.envs_tasks_desc import ENV_DESC, ENV_DIM, ENV_SAMPLE_NUM, env_mapper, TASK_SAMPLE_NUM, TASK_DESC, TASK_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Task:
    """Represents a task with unique identifier and associated skills"""
    uuid: TASK_UUID
    desc: str
    task_constant: Any
    skill_uuids: List[SKILL_UUID]
    label = "Task"

    @property
    def details(self):
        """Convert task constants to numpy array with fixed dimension"""
        dtl = np.hstack(self.task_constant)
        assert dtl.shape == (TASK_DIM, )
        return dtl

@dataclass
class Env:
    """Represents an environment with boundary and length properties"""
    uuid: ENV_UUID
    desc: str
    boundary: int
    length: Tuple[float, float]
    skill_uuids: List[SKILL_UUID]
    label = "Env"

    @property
    def details(self):
        """Combine boundary and length into a single array"""
        return np.hstack((self.boundary, self.length))

@dataclass
class Skill:
    """Represents a skill that connects environments and tasks"""
    uuid: SKILL_UUID
    desc: str
    env_uuids: List[ENV_UUID]
    task_uuids: List[TASK_UUID]
    skill_folder: str
    label = "Skill"

class SkillGraphBase:
    """Base class for managing skill graphs with environments, tasks, and skills"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        # Core data structures
        self.envs: Dict[ENV_UUID, Env] = {}
        self.raw_env_desc_index: Dict[str, List[ENV_UUID]] = {}  # env_name -> list of env uuids
        
        self.tasks: Dict[TASK_UUID, Task] = {}
        self.raw_task_desc_index: Dict[str, List[TASK_UUID]] = {}  # task_name -> list of task uuids
        
        self.skills: Dict[SKILL_UUID, Skill] = {}
        self.env_task_skill_index: Dict[Tuple[ENV_UUID, TASK_UUID], SKILL_UUID] = {}  # (env, task) -> skill mapping
        
        self.data_len = 0
        
        # Setup logging for training mode
        if not config["eval"]:
            self.log_dir = config["log_dir"]
            self.reporter = get_reporter(self.log_dir)

        # Model dimensions
        self.hidden_dim = config["hidden_dim"]
        self.skill_dim = config["skill_dim"]

        # Initialize encoders and optimizers
        self.env_encoder = Encoder(ENV_DIM, self.hidden_dim, self.skill_dim).to(DEVICE)
        self.env_optim = optim.AdamW(self.env_encoder.parameters(), lr=3e-4)
        
        self.task_encoder = Encoder(TASK_DIM, self.hidden_dim, self.skill_dim).to(DEVICE)
        self.task_optim = optim.AdamW(self.task_encoder.parameters(), lr=3e-4)

        # Load data and setup skill embeddings
        self._read(config['eval'])

        # Initialize skill embeddings
        self.skill_embeds = nn.Embedding(len(self.skills), self.skill_dim, device=DEVICE)
        normalize_embeds(self.skill_embeds)
        self.skill_optim = optim.AdamW(self.skill_embeds.parameters(), lr=3e-4)

        # Define objects to save during checkpointing
        self.save_obj = {
            "envs", "raw_env_desc_index", "tasks", "raw_task_desc_index",
            "skills", "env_task_skill_index", "data_len", "hidden_dim",
            "skill_dim", "skill_index_mapper", "max_env_delta_boundary",
            "max_env_delta_length", "max_task_delta_1", "max_task_delta_2"
        }
        
        # Create mapping from skill UUID to index
        self.skill_index_mapper = {suid: i for i, suid in enumerate(self.skills.keys())}

    def _read(self, eval: bool):
        """Load environments, tasks, and skills from configuration"""
        
        ######################### Add Environment #########################
        for env_name, props in tqdm(ENV_DESC.items(), desc="Add Environment"):
            boundary, _length = props
            
            # Handle variable length environments
            determinate_length = lambda idx: _length if not isinstance(_length, tuple) else np.random.uniform(low=_length[0], high=_length[1])

            # Create multiple environment instances
            for idx in range(ENV_SAMPLE_NUM):
                length = determinate_length(idx)
                env_uuid = self._add_env(dict(
                    desc=env_name,
                    boundary=boundary,
                    length=length
                ))
                
                # Update environment index
                if env_name not in self.raw_env_desc_index:
                    self.raw_env_desc_index[env_name] = [env_uuid]
                else:
                    self.raw_env_desc_index[env_name].append(env_uuid)

        # Compute normalization bounds for environments
        env_details = np.vstack([e.details for e in self.envs.values()])
        self.env_max = torch.as_tensor(env_details, dtype=torch.float32, device=DEVICE).max(dim=0).values + 1e-6
        self.env_min = torch.as_tensor(env_details, dtype=torch.float32, device=DEVICE).min(dim=0).values - 1e-6

        # Compute maximum deltas for training mode
        if not eval:
            e1 = np.repeat(env_details, len(self.envs), axis=0)
            e2 = np.vstack((env_details, ) * len(self.envs))
            assert len(e1.shape) == 2 == len(e2.shape)
            
            self.max_env_delta_boundary = np.max(np.linalg.norm(self.norm_env(e1)[:, [0]] - self.norm_env(e2)[:, [0]], axis=-1))
            print(f"The value of 'max_env_delta_boundary' is: {self.max_env_delta_boundary}")
            
            self.max_env_delta_length = np.max(np.linalg.norm(self.norm_env(e1)[:, [1]] - self.norm_env(e2)[:, [1]], axis=-1))
            print(f"The maximum value of 'max_env_delta_length' is: {self.max_env_delta_length}")
            
            del env_details, e1, e2

        print(f"There are {len(self.envs)} environments have been added.")

        ######################### Add Task #########################
        for task_name, props in tqdm(TASK_DESC.items(), desc="Add Task"):
            _v_max, _v_min, _topo_max, _d_ref, _d_sen = props
            
            # Handle variable task parameters
            determinate_v_max = lambda idx: _v_max if not isinstance(_v_max, tuple) else np.random.uniform(low=_v_max[0], high=_v_max[1])
            determinate_v_min = lambda idx: _v_min if not isinstance(_v_min, tuple) else np.random.uniform(low=_v_min[0], high=_v_min[1])
            determinate_topo_max = lambda idx: _topo_max if not isinstance(_topo_max, tuple) else np.random.uniform(low=_topo_max[0], high=_topo_max[1])
            determinate_d_ref = lambda idx: _d_ref if not isinstance(_d_ref, tuple) else np.random.uniform(low=_d_ref[0], high=_d_ref[1])
            determinate_d_sen = lambda idx: _d_sen if not isinstance(_d_sen, tuple) else np.random.uniform(low=_d_sen[0], high=_d_sen[1])

            # Create multiple task instances
            for idx in range(TASK_SAMPLE_NUM):
                v_max = determinate_v_max(idx)
                v_min = determinate_v_min(idx)
                topo_max = determinate_topo_max(idx)
                d_ref = determinate_d_ref(idx)
                d_sen = determinate_d_sen(idx)
                
                task_uuid = self._add_task(dict(
                    desc=task_name,
                    task_constant=(v_max, v_min, topo_max, d_ref, d_sen)
                ))
                
                # Update task index
                if task_name not in self.raw_task_desc_index:
                    self.raw_task_desc_index[task_name] = [task_uuid]
                else:
                    self.raw_task_desc_index[task_name].append(task_uuid)

        # Compute normalization bounds for tasks
        task_details = np.vstack([t.details for t in self.tasks.values()])
        self.task_max = torch.as_tensor(task_details, dtype=torch.float32, device=DEVICE).max(dim=0).values + 1e-6
        self.task_min = torch.as_tensor(task_details, dtype=torch.float32, device=DEVICE).min(dim=0).values - 1e-6

        # Compute maximum deltas for training mode
        if not eval:
            t1 = np.repeat(task_details, len(self.tasks), axis=0)
            t2 = np.vstack((task_details, ) * len(self.tasks))
            assert len(t1.shape) == 2 == len(t2.shape)
            
            self.max_task_delta_1 = np.max(np.linalg.norm(self.norm_task(t1)[:, [3]] - self.norm_task(t2)[:, [3]], axis=-1))
            print(f"The maximum of 'max_task_delta_1' is: {self.max_task_delta_1}")
            
            self.max_task_delta_2 = np.max(np.linalg.norm(self.norm_task(t1)[:, [4]] - self.norm_task(t2)[:, [4]], axis=-1))
            print(f"The maximum of 'max_task_delta_2' is: {self.max_task_delta_2}")
            
            del task_details, t1, t2

        print(f"There are {len(self.tasks)} tasks have been added.")

        ######################### Add Skill #########################
        # Create skills for all environment-task combinations
        for (env_desc, _), (task_desc, _) in product(ENV_DESC.items(), TASK_DESC.items()):
            env_uuids = self.raw_env_desc_index[env_desc]
            task_uuids = self.raw_task_desc_index[task_desc]
            self._add_skill(dict(
                env_uuids=env_uuids,
                task_uuids=task_uuids,
                env_desc=env_desc,
                task_desc=task_desc,
                env_id=env_mapper[env_desc],
            ))

        print("All skills have been added.")

    def _add_env(self, env_info: Dict[str, Any]) -> ENV_UUID:
        """Create and add a new environment"""
        uuid = gen_uuid()
        self.envs[uuid] = Env(
            uuid=uuid,
            desc=env_info["desc"],
            boundary=env_info["boundary"],
            length=env_info["length"],
            skill_uuids=[]
        )
        return uuid

    def _add_task(self, task_info: Dict[str, Any]) -> TASK_UUID:
        """Create and add a new task"""
        uuid = gen_uuid()
        self.tasks[uuid] = Task(
            uuid=uuid,
            desc=task_info["desc"],
            task_constant=task_info["task_constant"],
            skill_uuids=[]
        )
        return uuid

    def _add_skill(self, skill_info: Dict[str, Any]) -> SKILL_UUID:
        """Create and add a new skill, linking it to environments and tasks"""
        env_uuids = skill_info["env_uuids"]
        task_uuids = skill_info["task_uuids"]
        env_desc = skill_info["env_desc"]
        task_desc = skill_info["task_desc"]
        env_id = skill_info['env_id']
        
        skill_uuid = gen_uuid()
        skill = Skill(
            uuid=skill_uuid,
            desc=f"{task_desc}_{env_desc}",
            env_uuids=env_uuids,
            task_uuids=task_uuids,
            skill_folder=f'{task_desc}_{env_desc}'
        )
        self.skills[skill_uuid] = skill
        
        # Link skill to tasks
        for task_uuid in task_uuids:
            task = self.tasks[task_uuid]
            task.skill_uuids.append(skill_uuid)

        # Link skill to environments
        for env_uuid in env_uuids:
            env = self.envs[env_uuid]
            env.skill_uuids.append(skill_uuid)

        # Create environment-task-skill mapping
        for env_uuid, task_uuid in product(env_uuids, task_uuids):
            self.env_task_skill_index[(env_uuid, task_uuid)] = skill_uuid

        return skill_uuid

    def norm_env(self, envs: np.ndarray):
        """Normalize environment features to [0, 1] range"""
        return (envs - self.env_min.detach().cpu().numpy()) / (self.env_max.detach().cpu().numpy() - self.env_min.detach().cpu().numpy())

    def norm_task(self, tasks: np.ndarray):
        """Normalize task features to [0, 1] range"""
        return (tasks - self.task_min.detach().cpu().numpy()) / (self.task_max.detach().cpu().numpy() - self.task_min.detach().cpu().numpy())

    def save(self, counter: int, folder: Optional[str] = None):
        """Save model states to disk"""
        path = folder
        torch.save(self.env_encoder.state_dict(), f"{path}/env_encoder.pt")
        torch.save(self.task_encoder.state_dict(), f"{path}/task_encoder.pt")
        torch.save(self.skill_embeds.state_dict(), f"{path}/skill_embeds.pt")

    def load(self, folder):
        """Load model states from disk"""
        print(f"Load skill_graph model from '{folder}'")
        self.env_encoder.load_state_dict(torch.load(f"{folder}/env_encoder.pt"))
        self.task_encoder.load_state_dict(torch.load(f"{folder}/task_encoder.pt"))
        self.skill_embeds.load_state_dict(torch.load(f"{folder}/skill_embeds.pt"))
        self.kgc_trained = True

    def _pickle_save(self, path, obj):
        """Save object using pickle serialization"""
        with open(path, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _pickle_load(self, path):
        """Load object using pickle deserialization"""
        with open(path, "rb") as handle:
            return pickle.load(handle)