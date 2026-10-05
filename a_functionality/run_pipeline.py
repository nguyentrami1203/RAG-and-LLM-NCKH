import subprocess
import sys


STEPS = [

    "src/extract_axtree.py",

    "src/validate_benchmark.py",

    "src/generate_trajectories.py",

    "src/evaluate.py",

    "src/metrics.py",

    "src/offline_eval.py"
]


def main():

    print(
        "=" * 60
    )

    print(
        "AMAZON-BENCH MINI PIPELINE"
    )

    print(
        "=" * 60
    )


    for step in STEPS:

        print()

        print(
            f"Running: {step}"
        )

        print(
            "-" * 60
        )


        result = subprocess.run(
            [
                sys.executable,
                step
            ]
        )


        if result.returncode != 0:

            print()

            print(
                f"PIPELINE FAILED AT: "
                f"{step}"
            )

            sys.exit(1)


    print()

    print(
        "=" * 60
    )

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()