"""Play one complete random game and print the board after each turn."""
import sys
sys.path.insert(0, "src")

from backgammon_rlx.env.env import BackgammonEnv
from backgammon_rlx.agents.random_agent import RandomLegalAgent
from backgammon_rlx.notation.move_notation import format_full_turn

env    = BackgammonEnv()
agent0 = RandomLegalAgent(seed=0)
agent1 = RandomLegalAgent(seed=1)

obs = env.reset(seed=42)
step = 0
while not env.is_terminal():
    state  = env.state
    player = state.current_player
    turns  = env.legal_actions()
    agent  = agent0 if player == 0 else agent1
    action = agent.select_action(state, turns)
    print(f"Turn {step+1}  Player {player}  Dice {state.dice}  → {format_full_turn(action)}")
    obs, reward, done, info = env.step(action)
    step += 1
    if step % 20 == 0:
        print(env.render())
        print()

print(env.render())
print(f"\nGame over after {step} turns.")
print(f"Winner: player {info['winner']}, score: {info['score']}")
