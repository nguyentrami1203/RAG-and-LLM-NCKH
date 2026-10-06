import json
from pathlib import Path


INPUT_FILE = Path(
    "data/benchmark/offline_examples.jsonl"
)


def main():

    examples = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:

                examples.append(
                    json.loads(line)
                )


    correct = 0


    for example in examples:

        gold = example[
            "gold_action"
        ]

        predicted = example[
            "predicted_action"
        ]


        match = (
            gold == predicted
        )


        if match:

            correct += 1


        print(
            example["example_id"],
            "PASS"
            if match
            else "FAIL"
        )


    accuracy = (
        correct / len(examples)
        if examples
        else 0
    )


    print()

    print(
        f"Exact Match Accuracy: "
        f"{accuracy:.2%}"
    )


if __name__ == "__main__":
    main()