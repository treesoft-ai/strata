import sys
import time
from src.config import MODELS_DIR
from src.models.downloader import download_model
from src.models.manager import ModelManager


def main() -> None:
    """
    command line entrypoint for strata.
    supports:
    - uv run main.py download {huggingface-link}
    - uv run main.py run "prompt"
    - uv run main.py --model {model_name} run "prompt"
    """
    args = sys.argv[1:]
    model_name = None

    if len(args) >= 2 and args[0] == "--model":
        model_name = args[1]
        args = args[2:]

    if len(args) < 2:
        print("usage: uv run main.py [--model {model_name}] download {link} | run {prompt}")
        sys.exit(1)

    command: str = args[0].lower()
    argument: str = args[1]

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
            # load the model to inspect its running device
            active_model_name = model_name
            if not active_model_name:
                models = manager.get_available_models()
                if not models:
                    raise RuntimeError("no models have been downloaded yet. use download command first.")
                active_model_name = models[0]["name"]

            harness = manager.load_model(active_model_name)
            device = harness.device
            print(f"\n* strata / {device.lower()}\n")

            response_chunks = []
            start_time = time.time()
            # retrieve generated response stream
            for chunk in manager.generate_response_stream(argument):
                # print output strictly in lowercase (fully white, no caps UI)
                print(chunk.lower(), end="", flush=True)
                response_chunks.append(chunk)
            print()

            time_took = time.time() - start_time
            full_response = "".join(response_chunks)
            token_count = manager.count_tokens(full_response)
            tokens_per_second = token_count / time_took if time_took > 0 else 0.0

            # print stats surrounded by one empty line up and down
            print(f"\n* {time_took:.2f}s / {token_count} tokens / {tokens_per_second:.2f} tps\n")
        except Exception as err:
            print(f"\nerror: {str(err).lower()}")
            sys.exit(1)
    else:
        print(f"unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
