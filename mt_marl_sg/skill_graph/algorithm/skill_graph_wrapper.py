from typing import (Any, Callable, Dict, Optional, cast)
import numpy as np
import torch
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
# from neo4j import GraphDatabase
import random
from itertools import product

from skill_graph.algorithm import (
    Dataset, normalize_embeds, orth_to, 
    ENV_DIM, ENV_DESC, ENV_SAMPLE_NUM, TASK_DIM, TASK_DESC, TASK_SAMPLE_NUM, 
    SkillGraphBase, Env, Task, Skill
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORE_FACTOR = 3

def transH(h, w, d, t):
    """TransH scoring function for knowledge graph embeddings"""
    assert len(h.shape) == 2 == len(w.shape) == len(d.shape) == len(t.shape)
    batch_size = h.size(0)
    embed_dim = h.size(1)
    _h = h
    _w = w / torch.linalg.norm(w, dim=-1, keepdim=True) # unit normal vector
    _d = d
    _t = t

    # Project entities to relation-specific hyperplane: (h - w^T * h * w) + d - (t - w^T * t * w)
    scores = ((_h - (_w * _h).sum(-1, keepdim=True) * _w) + _d - (_t - (_w * _t).sum(-1, keepdim=True) * _w))

    assert scores.shape == (batch_size, embed_dim)
    # Calculate L2 norm as distance score
    scores = torch.linalg.norm(scores, dim=-1)

    # Convert distance to similarity score using exponential decay
    scores = torch.exp(-SCORE_FACTOR * scores)
    assert scores.shape == (batch_size, )

    return scores

class SkillGraph(SkillGraphBase):
    def __init__(self, config: Dict[str, Any]):
        super(SkillGraph, self).__init__(config)
        
        # Environment relation embeddings
        # Normal vector embedding - w is the unit normal vector of relation plane W_r
        self.r_w_e = nn.Embedding(len(self.skills), self.skill_dim).to(DEVICE)
        normalize_embeds(self.r_w_e)
        self.r_w_e_optim = optim.AdamW(self.r_w_e.parameters(), lr=3e-4)

        # Translation vector embedding - d is the translation vector of relation plane W_r
        self.r_d_e = nn.Embedding(len(self.skills), self.skill_dim).to(DEVICE)
        normalize_embeds(self.r_d_e)
        # Ensure translation vectors are orthogonal to normal vectors
        with torch.no_grad():
            for i in range(self.r_d_e.weight.size(0)):
                self.r_d_e.weight[i] = orth_to(self.r_w_e.weight[i], self.r_d_e.weight[i])

        self.r_d_e_optim = optim.AdamW(self.r_d_e.parameters(), lr=3e-4)

        # Task relation embeddings
        self.r_w_t = nn.Embedding(len(self.skills), self.skill_dim).to(DEVICE)
        normalize_embeds(self.r_w_t)
        self.r_w_t_optim = optim.AdamW(self.r_w_t.parameters(), lr=3e-4)

        self.r_d_t = nn.Embedding(len(self.skills), self.skill_dim).to(DEVICE)
        normalize_embeds(self.r_d_t)
        # Ensure translation vectors are orthogonal to normal vectors for tasks
        with torch.no_grad():
            for i in range(self.r_d_t.weight.size(0)):
                self.r_d_t.weight[i] = orth_to(self.r_w_t.weight[i], self.r_d_t.weight[i])

        self.r_d_t_optim = optim.AdamW(self.r_d_t.parameters(), lr=3e-4)

    def calc_env_margins(self):
        """Calculate pairwise margins between all environments"""
        assert not hasattr(self, "env_margins")
        env_margins = {}

        for e1, e2 in tqdm(product(self.envs.values(), self.envs.values()), desc="Calculate Env. Margins", total=len(self.envs)**2):
            e1d = self.norm_env(e1.details)
            e2d = self.norm_env(e2.details)
            assert e1d.shape == (ENV_DIM, )

            # Calculate weighted distance between environments
            bounadry_dist = np.linalg.norm(e1d[0] - e2d[0]) / self.max_env_delta_boundary
            length_dist = np.abs(e1d[1] - e2d[1]) / self.max_env_delta_length
            weight_1 = 0.95
            weight_2 = 1 - weight_1
            margin = weight_1 * bounadry_dist + weight_2 * length_dist
            # margin = max(length_dist, bounadry_dist)
            assert 0 <= margin <= 1

            if e1.uuid in env_margins:
                env_margins[e1.uuid][e2.uuid] = margin
            else:
                env_margins[e1.uuid] = {e2.uuid: margin}
        self.env_margins = env_margins

    def calc_task_margins(self):
        """Calculate pairwise margins between all tasks"""
        assert not hasattr(self, "task_margins")
        task_margins = {}

        for t1, t2 in tqdm(product(self.tasks.values(), self.tasks.values()), desc="Calculate Task Margins", total=len(self.tasks)**2):
            t1d = self.norm_task(t1.details)
            t2d = self.norm_task(t2.details)
            assert t1d.shape == (TASK_DIM, ) == t2d.shape

            # Calculate component-wise differences
            vel_diff = (np.abs(t1d[0] - t2d[0]) + np.abs(t1d[1] - t2d[1])) / 4
            task_diff = np.abs(t1d[2] - t2d[2]) / 2
            d_ref_diff = np.abs(t1d[3] - t2d[3]) / 2 
            d_sen_diff = np.abs(t1d[4] - t2d[4]) / 2

            # Weighted combination of different task features
            weight_vel = 0
            weight_nei = 15 
            weight_dref = 3 
            weight_dsen = 5 
            margin = (weight_vel * vel_diff + weight_nei * task_diff + weight_dref * d_ref_diff + weight_dsen * d_sen_diff) / (weight_vel + weight_nei + weight_dref + weight_dsen)
            assert 0 <= margin <= 1
            if t1.uuid in task_margins:
                task_margins[t1.uuid][t2.uuid] = margin
            else:
                task_margins[t1.uuid] = {t2.uuid: margin}

        self.task_margins = task_margins

    def train_skill_graph(self, config: Dict[str, Any]):
        """Train the skill graph using TransH knowledge graph completion"""
        self.calc_env_margins()
        self.calc_task_margins()
        batch_size, train_iters = (config["batch_size"], config["train_iters"])
        self.optims = [self.env_optim, self.task_optim, self.skill_optim, 
                    self.r_w_e_optim, self.r_d_e_optim, self.r_w_t_optim, self.r_d_t_optim]

        def assemble_env_tensor(e: Env):
            """Convert environment to normalized tensor"""
            return torch.as_tensor(self.norm_env(e.details),
                                dtype=torch.float32,
                                device=DEVICE)

        def assemble_task_tensor(t: Task):
            """Convert task to normalized tensor"""
            return torch.as_tensor(self.norm_task(t.details),
                                dtype=torch.float32,
                                device=DEVICE)

        # Pre-compute all environment and task tensors
        env_tensors = {
            euid: assemble_env_tensor(env)
            for euid, env in self.envs.items()
        }
        task_tensors = {
            tuid: assemble_task_tensor(task)
            for tuid, task in self.tasks.items()
        }

        def assemble_dataset():
            """Assemble positive and negative training triplets"""
            posi_env_data, posi_task_data = Dataset(), Dataset()
            
            # Positive triplets: (env, relation, skill) where env actually contains skill
            positive_envs = [(env_tensors[env.uuid], 
                            torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE), 
                            torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE),
                            torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE), 
                            (env, sid))
                            for env in self.envs.values()
                            for sid in env.skill_uuids]

            posi_env_data.add((torch.stack([e for (e, _, _, _, _) in positive_envs]),
                    torch.stack([w for (_, w, _, _, _) in positive_envs]),
                    torch.stack([d for (_, _, d, _, _) in positive_envs]),
                    torch.stack([s for (_, _, _, s, _) in positive_envs]), 
                    [dict(kind='e_r_s', env=i[0], sid=i[1]) for (_, _, _, _, i) in positive_envs]))

            # Positive triplets: (task, relation, skill) where task actually contains skill
            positive_tasks = [(task_tensors[task.uuid], 
                            torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE), 
                            torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE),
                            torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE), 
                            (task, sid))
                            for task in self.tasks.values()
                            for sid in task.skill_uuids]

            posi_task_data.add((torch.stack([t for (t, _, _, _, _) in positive_tasks]),
                    torch.stack([w for (_, w, _, _, _) in positive_tasks]),
                    torch.stack([d for (_, _, d, _, _) in positive_tasks]),
                    torch.stack([s for (_, _, _, s, _) in positive_tasks]), 
                    [dict(kind='t_r_s', task=i[0], sid=i[1]) for (_, _, _, _, i) in positive_tasks]))

            assert posi_task_data.len() + posi_env_data.len() == len(self.envs) * len(self.raw_task_desc_index) + len(self.tasks) * len(self.raw_env_desc_index)

            # Negative triplets for different violation types
            neg_env_wr_skill, neg_task_wr_skill, neg_env_r_env, neg_task_r_task = Dataset(), Dataset(), Dataset(), Dataset()

            # Type 1: env with wrong relation type (should use task relation)
            env_wr_skill = [(env_tensors[env.uuid], 
                    torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE), 
                    torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE),
                    torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE))
                    for env in self.envs.values() 
                    for sid in env.skill_uuids]

            neg_env_wr_skill.add((torch.stack([e for (e, _, _, _) in env_wr_skill]),
                            torch.stack([w for (_, w, _, _) in env_wr_skill]),
                            torch.stack([d for (_, _, d, _) in env_wr_skill]),
                            torch.stack([s for (_, _, _, s) in env_wr_skill]),
                            [dict(kind='e_wr_s') for _ in range(len(env_wr_skill))]))

            # Type 2: task with wrong relation type (should use env relation)
            task_wr_skill = [(task_tensors[task.uuid], 
                    torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE), 
                    torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE),
                    torch.tensor(self.skill_index_mapper[sid], dtype=torch.long, device=DEVICE))
                    for task in self.tasks.values()
                    for sid in task.skill_uuids]

            neg_task_wr_skill.add((torch.stack([e for (e, _, _, _) in task_wr_skill]),
                            torch.stack([w for (_, w, _, _) in task_wr_skill]),
                            torch.stack([d for (_, _, d, _) in task_wr_skill]),
                            torch.stack([s for (_, _, _, s) in task_wr_skill]),
                            [dict(kind='t_wr_s') for _ in range(len(task_wr_skill))]))

            # Type 3: env to env relations (should not exist)
            env_r_env = [(env_tensors[env1.uuid], 
                    torch.tensor(np.random.randint(0, len(self.skills)), dtype=torch.long, device=DEVICE), 
                    torch.tensor(np.random.randint(0, len(self.skills)), dtype=torch.long, device=DEVICE), 
                    env_tensors[env2.uuid])
                    for env1, env2 in tqdm(product(
                        random.sample(list(self.envs.values()), k=int(len(self.envs.values()))),
                        random.sample(list(self.envs.values()), k=int(len(self.envs.values())))), 
                        desc="Construct negative triples: env. -> env.")]

            neg_env_r_env.add((torch.stack([e for (e, _, _, _) in env_r_env]),
                        torch.stack([w for (_, w, _, _) in env_r_env]),
                        torch.stack([d for (_, _, d, _) in env_r_env]),
                        torch.stack([e for (_, _, _, e) in env_r_env]),
                        [dict(kind='e_r_e') for _ in range(len(env_r_env))]))

            # Type 4: task to task relations (should not exist)
            task_r_task = [(task_tensors[task1.uuid], 
                    torch.tensor(np.random.randint(0, len(self.skills)), dtype=torch.long, device=DEVICE), 
                    torch.tensor(np.random.randint(0, len(self.skills)), dtype=torch.long, device=DEVICE), 
                    task_tensors[task2.uuid])
                    for task1, task2 in tqdm(product(
                        random.sample(list(self.tasks.values()), k=int(len(self.tasks.values()))),
                        random.sample(list(self.tasks.values()), k=int(len(self.tasks.values())))), 
                        desc="Construct negative triples: task -> task")]

            neg_task_r_task.add((torch.stack([t for (t, _, _, _) in task_r_task]),
                        torch.stack([w for (_, w, _, _) in task_r_task]),
                        torch.stack([d for (_, _, d, _) in task_r_task]),
                        torch.stack([t for (_, _, _, t) in task_r_task]),
                        [dict(kind='t_r_t') for _ in range(len(task_r_task))]))

            return (posi_env_data, posi_task_data), (neg_env_wr_skill, neg_task_wr_skill, neg_env_r_env, neg_task_r_task)

        # Assemble training data
        (posi_env_data, posi_task_data), (neg_env_wr_skill, neg_task_wr_skill, neg_env_r_env, neg_task_r_task) = assemble_dataset()
        
        # Define loss functions
        rank_loss_fn = nn.ReLU()
        score_loss_fn = nn.MSELoss()
        entity_loss_fn = nn.ReLU()
        orth_loss_fn = nn.ReLU()
        orth_epsilon = 1e-4

        cnt = 0
        num_epoch = 10
        
        # Main training loop
        for _ in tqdm(range(train_iters)):
            self.env_encoder.train()
            self.task_encoder.train()
            
            for _ in range(num_epoch):
                # Sample positive triplets
                p_e_r_s, p_t_r_s = posi_env_data.sample(batch_size), posi_task_data.sample(batch_size)

                # Calculate scores for positive triplets
                correct_scores = torch.cat(
                    (transH(
                        self.env_encoder(p_e_r_s[0]),
                        self.r_w_e(p_e_r_s[1]),
                        self.r_d_e(p_e_r_s[2]),
                        self.skill_embeds(p_e_r_s[3])),
                    transH(
                        self.task_encoder(p_t_r_s[0]),
                        self.r_w_t(p_t_r_s[1]),
                        self.r_d_t(p_t_r_s[2]),
                        self.skill_embeds(p_t_r_s[3]))
                    ))

                # Generate soft negative triplets with margin-based ranking
                _wrong_task_triplets, _wrong_env_triplets = [], []

                # For each positive env-skill pair, find a negative env not containing the skill
                for i in p_e_r_s[-1]:
                    assert i['kind'] == 'e_r_s'
                    o_env = cast(Env, i['env'])
                    skill = cast(Skill, self.skills[i['sid']])

                    assert all(map(lambda eid: eid in self.envs.keys(), skill.env_uuids))

                    # Choose env that doesn't contain this skill
                    candidates = set(self.envs.keys()).difference(*[skill.env_uuids])
                    n_env_id = np.random.choice(list(candidates))

                    _wrong_env_triplets.append((env_tensors[n_env_id], 
                                                torch.tensor(self.skill_index_mapper[skill.uuid], dtype=torch.long, device=DEVICE), 
                                                torch.tensor(self.skill_index_mapper[skill.uuid], dtype=torch.long, device=DEVICE),
                                                torch.tensor(self.skill_index_mapper[skill.uuid], dtype=torch.long, device=DEVICE),
                                                self.env_margins[o_env.uuid][n_env_id]))

                # For each positive task-skill pair, find a negative task not containing the skill
                for i in p_t_r_s[-1]:
                    assert i['kind'] == 't_r_s'
                    o_task = cast(Task, i['task'])
                    skill = cast(Skill, self.skills[i['sid']])

                    assert all(map(lambda tid: tid in self.tasks.keys(), skill.task_uuids))

                    # Choose task that doesn't contain this skill
                    candidates = set(self.tasks.keys()).difference(*[skill.task_uuids])
                    n_task_id = np.random.choice(list(candidates))

                    _wrong_task_triplets.append((task_tensors[n_task_id], 
                                                torch.tensor(self.skill_index_mapper[skill.uuid], dtype=torch.long, device=DEVICE), 
                                                torch.tensor(self.skill_index_mapper[skill.uuid], dtype=torch.long, device=DEVICE),
                                                torch.tensor(self.skill_index_mapper[skill.uuid], dtype=torch.long, device=DEVICE),
                                                self.task_margins[o_task.uuid][n_task_id]))

                # Stack wrong triplets into tensors
                wrong_env_triplets = [
                    torch.stack([e for (e, _, _, _, _) in _wrong_env_triplets]),
                    torch.stack([w for (_, w, _, _, _) in _wrong_env_triplets]),
                    torch.stack([d for (_, _, d, _, _) in _wrong_env_triplets]),
                    torch.stack([s for (_, _, _, s, _) in _wrong_env_triplets]),
                    torch.tensor([m for (_, _, _, _, m) in _wrong_env_triplets], dtype=torch.float32, device=DEVICE)
                ]
                wrong_task_triplets = [
                    torch.stack([t for (t, _, _, _, _) in _wrong_task_triplets]),
                    torch.stack([w for (_, w, _, _, _) in _wrong_task_triplets]),
                    torch.stack([d for (_, _, d, _, _) in _wrong_task_triplets]),
                    torch.stack([s for (_, _, _, s, _) in _wrong_task_triplets]),
                    torch.tensor([m for (_, _, _, _, m) in _wrong_task_triplets], dtype=torch.float32, device=DEVICE)
                ]

                # Calculate scores for soft triplets
                (weh, wew, wed, wet, wem) = wrong_env_triplets
                (wth, wtw, wtd, wtt, wtm) = wrong_task_triplets
                wrong_scores = torch.cat(
                    (transH(self.env_encoder(weh), 
                            self.r_w_e(wew),
                            self.r_d_e(wed), 
                            self.skill_embeds(wet)),
                    transH(self.task_encoder(wth), 
                            self.r_w_t(wtw),
                            self.r_d_t(wtd), 
                            self.skill_embeds(wtt))
                    ))

                # Margin-based ranking loss
                margins = torch.cat((wem, wtm))
                ranking_loss = rank_loss_fn(-(torch.ones_like(wrong_scores, device=DEVICE) - wrong_scores - margins.detach())).mean()

                # Sample hard negative triplets
                newrs, ntwrs, nere, ntrt = (neg_env_wr_skill.sample(batch_size), 
                                        neg_task_wr_skill.sample(batch_size), 
                                        neg_env_r_env.sample(batch_size), 
                                        neg_task_r_task.sample(batch_size))

                # Calculate scores for soft negative triplets
                negative_scores = torch.cat(
                    (transH(self.env_encoder(newrs[0]),
                            self.r_w_t(newrs[1]),  # Wrong relation type
                            self.r_d_t(newrs[2]),
                            self.skill_embeds(newrs[3])),
                    transH(self.task_encoder(ntwrs[0]),
                            self.r_w_e(ntwrs[1]),  # Wrong relation type
                            self.r_d_e(ntwrs[2]),
                            self.skill_embeds(ntwrs[3])),
                    transH(self.env_encoder(nere[0]), 
                            self.r_w_e(nere[1]),
                            self.r_d_e(nere[2]),
                            self.env_encoder(nere[3])),  # Invalid env-env relation
                    transH(self.task_encoder(ntrt[0]),
                            self.r_w_t(ntrt[1]), 
                            self.r_d_t(ntrt[2]),
                            self.task_encoder(ntrt[3]))  # Invalid task-task relation
                    ))

                # Binary classification loss: positive triplets should score 1, negative should score 0
                score_loss = (score_loss_fn(negative_scores, torch.zeros_like(negative_scores)) + 
                            score_loss_fn(correct_scores, torch.ones_like(correct_scores)))

                # Entity embedding regularization: all entities should have unit norm
                env_entity_loss = entity_loss_fn(
                    torch.linalg.norm(self.env_encoder(torch.cat([p_e_r_s[0], weh, newrs[0], nere[0], nere[3]])), dim=-1) - 1).mean()

                task_entity_loss = entity_loss_fn(
                    torch.linalg.norm(self.task_encoder(torch.cat([p_t_r_s[0], wth, ntwrs[0], ntrt[0], ntrt[3]])), dim=-1) - 1).mean()

                skill_entity_loss = entity_loss_fn(
                    torch.linalg.norm(self.skill_embeds(torch.arange(len(self.skills), device=DEVICE, dtype=torch.long)), dim=-1) - 1).mean()

                # Orthogonality constraint: relation normal vectors should be orthogonal to translation vectors
                _w = self.r_w_e.weight
                _w = _w / torch.linalg.norm(_w, dim=-1, keepdim=True)
                orths_e = (_w * self.r_d_e.weight).sum(-1) / torch.linalg.norm(self.r_d_e.weight, dim=-1)

                _w = self.r_w_t.weight
                _w = _w / torch.linalg.norm(_w, dim=-1, keepdim=True)
                orths_t = (_w * self.r_d_t.weight).sum(-1) / torch.linalg.norm(self.r_d_t.weight, dim=-1)

                assert orths_e.shape == (self.r_w_e.weight.size(0), )
                assert orths_t.shape == (self.r_w_t.weight.size(0), )
                relation_orth_loss = (orth_loss_fn(orths_e**2 - orth_epsilon**2).mean() + 
                                    orth_loss_fn(orths_t**2 - orth_epsilon**2).mean())

                # Backward pass and optimization
                [o.zero_grad() for o in self.optims]
                total_loss = (1.5 * ranking_loss + 1.5 * score_loss + 
                            0.5 * (task_entity_loss + env_entity_loss + skill_entity_loss + relation_orth_loss))
                total_loss.backward()
                [o.step() for o in self.optims]

            self.env_encoder.eval()
            self.task_encoder.eval()
            
            # Logging and model saving
            cnt += 1
            if cnt % 15 == 0:
                self.save(cnt)
                self.reporter._writer.add_scalars('summary', {'total_loss': total_loss.item()}, cnt)

        # Close tensorboard writer
        self.reporter._writer.export_scalars_to_json(self.log_dir + '/summary/summary.json')
        self.reporter._writer.close()

        self.kgc_trained = True
        self.env_encoder.eval()
        self.task_encoder.eval()

    def kgc(self, env_property: np.ndarray, task_property: np.ndarray, 
            merge_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            reverse=False):
        """Knowledge Graph Completion: predict skills given environment and task properties"""
        assert hasattr(self, "kgc_trained") and self.kgc_trained
        skill_reverse_mapper = {i: suid for (suid, i) in self.skill_index_mapper.items()}

        merge_fn = merge_fn or (lambda es, ts: es * ts)

        # Normalize and convert query properties to tensors
        q_e = torch.as_tensor(self.norm_env(env_property), dtype=torch.float32, device=DEVICE)
        q_t = torch.as_tensor(self.norm_task(task_property), dtype=torch.float32, device=DEVICE)
        
        l = len(self.skills)
        with torch.no_grad():
            # Get all skill embeddings
            all_skills = self.skill_embeds(torch.arange(l, dtype=torch.long, device=DEVICE))

            # Calculate environment-skill compatibility scores
            es = self.env_encoder(q_e.unsqueeze(0)).repeat_interleave(l, dim=0)
            e_w_s = self.r_w_e(torch.arange(l, dtype=torch.long, device=DEVICE))
            e_d_s = self.r_d_e(torch.arange(l, dtype=torch.long, device=DEVICE))
            e_scores = transH(es, e_w_s, e_d_s, all_skills)
            assert e_scores.shape == (l, )

            # Calculate task-skill compatibility scores
            ts = self.task_encoder(q_t.unsqueeze(0)).repeat_interleave(l, dim=0)
            t_w_s = self.r_w_t(torch.arange(l, dtype=torch.long, device=DEVICE))
            t_d_s = self.r_d_t(torch.arange(l, dtype=torch.long, device=DEVICE))
            t_scores = transH(ts, t_w_s, t_d_s, all_skills)
            assert t_scores.shape == (l, )

        # Merge environment and task scores
        final_scores = merge_fn(e_scores, t_scores)
        assert final_scores.shape == (l, )
        idxs = final_scores.argsort(descending=True)

        return ([self.skills[skill_reverse_mapper[i.item()]] for i in idxs], 
                final_scores[idxs], 
                (e_scores[idxs], t_scores[idxs]))

    def draw_in_neo4j(self):
        """Visualize the skill graph in Neo4j database"""
        
        def add_env(tx, env: Env):
            query = "CREATE (e: Env {uuid: $uuid, skill_uuids: $skill_uuids, label: $label, desc: $desc})"
            tx.run(query, uuid=env.uuid, skill_uuids=env.skill_uuids, label=ENV_DESC[env.desc], desc=env.desc)

        def add_task(tx, task: Task):
            query = "CREATE (t: Task {uuid: $uuid, skill_uuids: $skill_uuids, label: $label, desc: $desc})"
            tx.run(query, uuid=task.uuid, skill_uuids=task.skill_uuids, label=TASK_DESC[task.desc], desc=task.desc)

        def add_skill(tx, skill: Skill):
            query = "CREATE (s: Skill {uuid: $skill_uuid, label: $label, env_uuids: $env_uuids, task_uuids: $task_uuids, desc: $desc})"
            tx.run(query, skill_uuid=skill.uuid, env_uuids=skill.env_uuids, desc=skill.desc, 
                task_uuids=skill.task_uuids, label=skill.label)

        def add_env_relations(tx, **kwargs):
            query = "MATCH (e: Env {uuid: $env_uuid}), (s: Skill {uuid: $skill_uuid}) CREATE (e) -[:R] -> (s)"
            tx.run(query, **kwargs)

        def add_task_relations(tx, **kwargs):
            query = "MATCH (t: Task {uuid: $task_uuid}), (s: Skill {uuid: $skill_uuid}) CREATE (t) -[:R] -> (s)"
            tx.run(query, **kwargs)

        # Connect to Neo4j and populate the graph
        with GraphDatabase.driver("neo4j+s://54f805e8.databases.neo4j.io", auth=("neo4j", "your_password")) as driver:
            with driver.session() as session:
                # Clear existing data
                session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

                # Add nodes
                for skill in list(self.skills.values())[:250]:
                    session.execute_write(add_skill, skill)

                for env in list(self.envs.values())[::ENV_SAMPLE_NUM]:
                    session.execute_write(add_env, env)

                for task in list(self.tasks.values())[::TASK_SAMPLE_NUM]:
                    session.execute_write(add_task, task)

                # Add relationships
                for skill in tqdm(list(self.skills.values())[:250], desc="Add relations"):
                    for env_uuid in skill.env_uuids:
                        session.execute_write(add_env_relations, skill_uuid=skill.uuid, env_uuid=env_uuid)
                    for task_uuid in skill.task_uuids:
                        session.execute_write(add_task_relations, skill_uuid=skill.uuid, task_uuid=task_uuid)

    def save(self, counter: int, folder: Optional[str] = None):
        """Save model state including TransH relation embeddings"""
        if folder is None:
            folder = f'{self.log_dir}/save/{counter}'
        path = folder
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save TransH relation embeddings
        torch.save(self.r_w_e.state_dict(), f"{path}/r_w_e.pt")
        torch.save(self.r_d_e.state_dict(), f"{path}/r_d_e.pt")
        torch.save(self.r_w_t.state_dict(), f"{path}/r_w_t.pt")
        torch.save(self.r_d_t.state_dict(), f"{path}/r_d_t.pt")
        super().save(counter, folder)

    def load(self, folder):
        """Load model state including TransH relation embeddings"""
        self.r_w_e.load_state_dict(torch.load(f"{folder}/r_w_e.pt"))
        self.r_d_e.load_state_dict(torch.load(f"{folder}/r_d_e.pt"))
        self.r_w_t.load_state_dict(torch.load(f"{folder}/r_w_t.pt"))
        self.r_d_t.load_state_dict(torch.load(f"{folder}/r_d_t.pt"))
        super().load(folder)
        self.env_encoder.eval()
        self.task_encoder.eval()