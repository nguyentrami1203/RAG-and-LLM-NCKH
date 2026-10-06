import json

from environment import (
    MiniAmazonEnvironment
)


with open(
    "data/benchmark/initial_state.json",
    "r",
    encoding="utf-8"
) as file:

    initial_state = json.load(file)


env = MiniAmazonEnvironment(
    initial_state
)


print("INITIAL STATE")

print(
    json.dumps(
        env.get_state(),
        indent=2
    )
)


env.select(
    "qty-sony",
    "2"
)

env.stop(
    "Task completed."
)


print("\nFINAL STATE")

print(
    json.dumps(
        env.get_state(),
        indent=2
    )
)


print("\nACTION HISTORY")

print(
    json.dumps(
        env.get_history(),
        indent=2
    )
)