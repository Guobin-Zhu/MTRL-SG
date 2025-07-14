################################# env description #################################
ENV_DESC = {
    'fixed': (1, (4.5, 5.5)),
    'periodic': (0, (4.5, 5.5)),
}

env_mapper = {k: i for i, k in enumerate(ENV_DESC.keys())}

ENV_NUM = len(ENV_DESC)
first_key = next(iter(ENV_DESC))
ENV_DIM = len(ENV_DESC[first_key])
ENV_SAMPLE_NUM = 50

################################# task description #################################
TASK_DESC = {
    ## version 1
    'flocking_0.4_2': (1.0, 0, 0, (1.5,2.5), (0.35,0.45)),
    'flocking_0.8_2': (1.0, 0, 0, (1.5,2.5), (0.75,0.85)),
    'flocking_0.4_3': (1.0, 0, 0, (2.5,3.5), (0.35,0.45)),
    'flocking_0.8_3': (1.0, 0, 0, (2.5,3.5), (0.75,0.85)),
    'flocking_0.4_4': (1.0, 0, 0, (3.5,4.5), (0.35,0.45)),
    'flocking_0.8_4': (1.0, 0, 0, (3.5,4.5), (0.75,0.85)),
    'flocking_0.4_5': (1.0, 0, 0, (4.5,5.5), (0.35,0.45)),
    'flocking_0.8_5': (1.0, 0, 0, (4.5,5.5), (0.75,0.85)),
    'adversarial_2_0.3': (1.0, 0, 1, (1.9,2.1), (0.29,0.31)),
    'adversarial_3_0.3': (1.0, 0, 1, (2.9,3.1), (0.29,0.31)),
    'adversarial_4_0.3': (1.0, 0, 1, (3.9,4.1), (0.29,0.31)),
    'adversarial_5_0.3': (1.0, 0, 1, (4.9,5.1), (0.29,0.31)),
    'adversarial_2_0.4': (1.0, 0, 1, (1.9,2.1), (0.39,0.41)),
    'adversarial_3_0.4': (1.0, 0, 1, (2.9,3.1), (0.39,0.41)),
    'adversarial_4_0.4': (1.0, 0, 1, (3.9,4.1), (0.39,0.41)),
    'adversarial_5_0.4': (1.0, 0, 1, (4.9,5.1), (0.39,0.41)),
}

TASK_SAMPLE_NUM = 50
TASK_DIM = 5

################################# uid #################################

i = -1
def gen_uuid() -> int:
    global i
    i += 1
    return i

UUID = int
TASK_UUID = UUID
ENV_UUID = UUID
SKILL_UUID = UUID