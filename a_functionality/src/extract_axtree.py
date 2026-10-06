from bs4 import BeautifulSoup
from pathlib import Path
import json


# Những HTML element chúng ta coi là
# thành phần tương tác với người dùng
INTERACTIVE_TAGS = [
    "button",
    "input",
    "select",
    "a"
]


def extract_axtree(html_content):

    soup = BeautifulSoup(
        html_content,
        "html.parser"
    )

    elements = []

    element_id = 1

    for tag in soup.find_all(INTERACTIVE_TAGS):

        # Xác định role đơn giản
        if tag.name == "button":
            role = "button"

        elif tag.name == "input":
            role = "textbox"

        elif tag.name == "select":
            role = "combobox"

        elif tag.name == "a":
            role = "link"

        else:
            role = tag.name

        # Lấy text hiển thị
        text = tag.get_text(
            " ",
            strip=True
        )

        # Nếu input không có text,
        # dùng placeholder
        if not text:
            text = tag.get(
                "placeholder",
                ""
            )

        element = {
            "ax_id": element_id,
            "role": role,
            "html_id": tag.get("id"),
            "name": text
        }

        # Nếu là select,
        # lấy các option
        if tag.name == "select":

            options = []

            for option in tag.find_all("option"):

                options.append(
                    option.get_text(
                        " ",
                        strip=True
                    )
                )

            element["options"] = options

        elements.append(element)

        element_id += 1

    return elements


def main():

    input_directory = Path(
        "data/pages"
    )

    output_directory = Path(
        "data/axtrees"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    html_files = list(
        input_directory.glob("*.html")
    )

    print(
        f"Found {len(html_files)} HTML pages."
    )

    for html_file in html_files:

        print(
            f"Processing: {html_file.name}"
        )

        html_content = html_file.read_text(
            encoding="utf-8"
        )

        axtree = extract_axtree(
            html_content
        )

        output_file = (
            output_directory
            / f"{html_file.stem}.json"
        )

        output_file.write_text(
            json.dumps(
                axtree,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print(
            f"Created: {output_file}"
        )


if __name__ == "__main__":
    main()