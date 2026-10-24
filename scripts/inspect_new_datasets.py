from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLAAD_PATH = PROJECT_ROOT / "dataset" / "raw" / "MLAAD-tiny"
LIBRISPEECH_PATH = PROJECT_ROOT / "dataset" / "raw" / "LibriSpeech"


def inspect_folder(name, path):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    if not path.exists():
        print("Folder does not exist:")
        print(path)
        return

    print("Location:")
    print(path)

    print("\nTop-level contents:")

    items = list(path.iterdir())

    for item in items[:30]:
        if item.is_dir():
            print("[DIR] ", item.name)
        else:
            print("[FILE]", item.name)

    print("\nTotal top-level items:", len(items))


def main():

    inspect_folder("MLAAD-TINY", MLAAD_PATH)

    inspect_folder("LIBRISPEECH", LIBRISPEECH_PATH)


if __name__ == "__main__":
    main()