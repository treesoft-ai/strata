import sys
from src.config import MODELS_DIR
from src.models.downloader import download_model
from src.models.manager import ModelManager


def main() -> None:
    """
    command line entrypoint for strata.
    supports:
    - uv run main.py download {huggingface-link}
    - uv run main.py run "prompt"
    """
    if len(sys.argv) < 3:
        print("usage: uv run main.py download {link} | run {prompt}")
        sys.exit(1)

    command: str = sys.argv[1].lower()
    argument: str = sys.argv[2]

    if command == "download":
        print(f"starting download for link: {argument.lower()}")
        try:
            download_model(
                link=argument,
                dest_dir=MODELS_DIR,
                progress_callback=lambda msg: print(msg.lower())
            )
            print("download finished successfully")
        except Exception as err:
            print(f"error: {str(err).lower()}")
            sys.exit(1)

    elif command == "run":
        manager = ModelManager()
        try:
            # retrieve generated response
            response = manager.generate_response(argument)
            # print output strictly in lowercase (fully white, no caps UI)
            print(response.lower())
        except Exception as err:
            print(f"error: {str(err).lower()}")
            sys.exit(1)
    else:
        print(f"unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
