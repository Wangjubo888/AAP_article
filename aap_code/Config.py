class Config:
    num_agents = 5
    R1 = 50  # 起降场半径
    R2 = 100  # 进近区域半径
    R3 = 200  # 空域范围半径
    state_dim = 12  # 状态维度
    action_dim = 4  # 动作维度
    lr = 0.0003  # 学习率
    gamma = 0.99  # 折扣因子
    epsilon = 0.2  # PPO裁剪参数
    max_episodes = 1000
    max_steps = 500
    batch_size = 64
    use_wandb = True  # 是否使用WandB进行实验记录
    landing_interval = 10  # 起降场使用的最小间隔时间，单位为秒

