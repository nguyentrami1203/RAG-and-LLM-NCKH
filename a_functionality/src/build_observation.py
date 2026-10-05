import json
from pathlib import Path

from actions import ACTION_SPACE


def build_observation(
    task,
    axtree,
    action_history
):

    return {

        "user_query":
            task["query"],

        "axtree":
            axtree,

        "action_space":
            ACTION_SPACE,

        "action_history":
            action_history
    }


if __name__ == "__main__":

    with open(
        "data/axtrees/cart.json",
        "r",
        encoding="utf-8"
    ) as file:

        axtree = json.load(file)


    task = {
        "query":
        "Change the quantity of the Sony headphones to 2."
    }


    observation = build_observation(
        task,
        axtree,
        []
    )


    print(
        json.dumps(
            observation,
            indent=2
        )
    )