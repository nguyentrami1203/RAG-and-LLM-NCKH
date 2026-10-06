import json
from pathlib import Path


TASK_FILE = Path(
    "data/benchmark/tasks.jsonl"
)

AXTREE_DIRECTORY = Path(
    "data/axtrees"
)


def load_tasks():

    tasks = []

    with TASK_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                tasks.append(
                    json.loads(line)
                )

    return tasks


def load_axtree(page_name):

    page_stem = Path(page_name).stem

    file_path = (
        AXTREE_DIRECTORY
        / f"{page_stem}.json"
    )

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def validate_task(task):

    axtree = load_axtree(
        task["source_page"]
    )

    available_ids = {
        element["html_id"]
        for element in axtree
        if element["html_id"]
    }

    target = task[
        "target_element"
    ]

    return target in available_ids


def main():

    tasks = load_tasks()

    passed = 0

    print(
        f"Validating {len(tasks)} tasks...\n"
    )

    for task in tasks:

        valid = validate_task(task)

        if valid:

            status = "PASS"
            passed += 1

        else:

            status = "FAIL"

        print(
            task["task_id"],
            status,
            "-",
            task["target_element"]
        )

    print()

    print(
        f"Passed: {passed}/{len(tasks)}"
    )


if __name__ == "__main__":
    main()