from gym.envs.registration import register


register(
    id='maze-v0',
    entry_point='gym_maze.envs:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-sample-5x5-v0',
    entry_point='gym_maze.envs:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-random-5x5-v0',
    entry_point='gym_maze.envs:MazeEnvRandom5x5',
    max_episode_steps=2000,
    nondeterministic=True,
)

register(
    id='maze-sample-10x10-v0',
    entry_point='gym_maze.envs:MazeEnvSample10x10',
    max_episode_steps=10000,
)

register(
    id='maze-random-10x10-v0',
    entry_point='gym_maze.envs:MazeEnvRandom10x10',
    max_episode_steps=10000,
    nondeterministic=True,
)

register(
    id='maze-sample-3x3-v0',
    entry_point='gym_maze.envs:MazeEnvSample3x3',
    max_episode_steps=1000,
)

register(
    id='maze-random-3x3-v0',
    entry_point='gym_maze.envs:MazeEnvRandom3x3',
    max_episode_steps=1000,
    nondeterministic=True,
)


register(
    id='maze-sample-100x100-v0',
    entry_point='gym_maze.envs:MazeEnvSample100x100',
    max_episode_steps=1000000,
)

register(
    id='maze-random-100x100-v0',
    entry_point='gym_maze.envs:MazeEnvRandom100x100',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-10x10-plus-v0',
    entry_point='gym_maze.envs:MazeEnvRandom10x10Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-20x20-plus-v0',
    entry_point='gym_maze.envs:MazeEnvRandom20x20Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-30x30-plus-v0',
    entry_point='gym_maze.envs:MazeEnvRandom30x30Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)


register(
    id='maze-sample-20x20-v0',
    entry_point='gym_maze.envs:MazeEnvSample20x20',
    max_episode_steps=50000,
    nondeterministic=True,
)

register(
    id='maze-sample-30x30-v0',
    entry_point='gym_maze.envs:MazeEnvSample30x30',
    max_episode_steps=50000,
    nondeterministic=True,
)

register(
    id='maze-sample-40x40-v0',
    entry_point='gym_maze.envs:MazeEnvSample40x40',
    max_episode_steps=50000,
    nondeterministic=True,
)

register(
    id='maze-sample-50x50-v0',
    entry_point='gym_maze.envs:MazeEnvSample50x50',
    max_episode_steps=75000,
    nondeterministic=True,
)

register(
    id='maze-sample-60x60-v0',
    entry_point='gym_maze.envs:MazeEnvSample60x60',
    max_episode_steps=100000,
    nondeterministic=True,
)

register(
    id='maze-sample-70x70-v0',
    entry_point='gym_maze.envs:MazeEnvSample70x70',
    max_episode_steps=100000,
    nondeterministic=True,
)

register(
    id='maze-sample-80x80-v0',
    entry_point='gym_maze.envs:MazeEnvSample80x80',
    max_episode_steps=100000,
    nondeterministic=True,
)

register(
    id='maze-sample-90x90-v0',
    entry_point='gym_maze.envs:MazeEnvSample90x90',
    max_episode_steps=100000,
    nondeterministic=True,
)

register(
    id='maze-sample-25x25-v0',
    entry_point='gym_maze.envs:MazeEnvSample25x25',
    max_episode_steps=100000,
)

register(
    id='maze-random-25x25-v0',
    entry_point='gym_maze.envs:MazeEnvRandom25x25',
    max_episode_steps=100000,
    nondeterministic=True,
)
