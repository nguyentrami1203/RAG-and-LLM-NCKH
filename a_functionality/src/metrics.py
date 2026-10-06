import json
from pathlib import Path


RESULT_FILE = Path(
    "results/evaluation_results.json"
)


def main():

    with RESULT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        results = json.load(file)


    total = len(results)


    success = sum(
        1
        for result in results
        if result["verdict"]
        == "SUCCESS"
    )


    benign = sum(
        1
        for result in results
        if result["verdict"]
        == "BENIGN_FAILURE"
    )


    harmful = sum(
        1
        for result in results
        if result["verdict"]
        == "HARMFUL_FAILURE"
    )


    metrics = {

        "total_trajectories":
            total,

        "success_count":
            success,

        "benign_failure_count":
            benign,

        "harmful_failure_count":
            harmful,

        "success_rate":
            success / total
            if total else 0,

        "benign_failure_rate":
            benign / total
            if total else 0,

        "harmful_failure_rate":
            harmful / total
            if total else 0
    }


    print(
        json.dumps(
            metrics,
            indent=2
        )
    )


    output_file = Path(
        "results/metrics.json"
    )

    output_file.write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()