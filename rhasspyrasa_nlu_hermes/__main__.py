"""Hermes MQTT service for Rasa NLU"""
import argparse
import asyncio
import logging
import typing
from pathlib import Path

import paho.mqtt.client as mqtt

from . import NluHermesMqtt

_LOGGER = logging.getLogger("rhasspyrasa_nlu_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    parser = argparse.ArgumentParser(prog="rhasspy-rasa-nlu-hermes")
    parser.add_argument("--rasa-url", required=True, help="URL of Rasa NLU server")
    parser.add_argument("--intent-graph", help="Path to rhasspy intent graph JSON file")
    parser.add_argument("--examples-path", help="Path to write examples markdown file")
    parser.add_argument("--rasa-config", help="Path to Rasa NLU's config.yml file")
    parser.add_argument(
        "--rasa-project",
        default="rhasspy",
        help="Project name used when training Rasa NLU (default: rhasspy)",
    )
    parser.add_argument(
        "--rasa-model-dir",
        default="models",
        help="Directory name where Rasa NLU stores its model files (default: models)",
    )
    parser.add_argument(
        "--rasa-language",
        default="en",
        help="Language used for Rasa NLU training (default: en)",
    )
    parser.add_argument(
        "--write-graph",
        action="store_true",
        help="Write training graph to intent-graph path",
    )
    parser.add_argument(
        "--casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for input text (default: ignore)",
    )
    parser.add_argument(
        "--replace-numbers",
        action="store_true",
        help="Replace digits with words in queries (75 -> seventy five)",
    )
    parser.add_argument(
        "--number-language",
        help="Language/locale used for number replacement (default: en)",
    )
    parser.add_argument("--certfile", help="SSL certificate file")
    parser.add_argument("--keyfile", help="SSL private key file (optional)")
    parser.add_argument(
        "--host", default="localhost", help="MQTT host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=1883, help="MQTT port (default: 1883)"
    )
    parser.add_argument(
        "--siteId",
        action="append",
        help="Hermes siteId(s) to listen for (default: all)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    try:
        loop = asyncio.get_event_loop()

        # Convert to Paths
        if args.intent_graph:
            args.intent_graph = Path(args.intent_graph)

        if args.examples_path:
            args.examples_path = Path(args.examples_path)

        if args.rasa_config:
            args.rasa_config = Path(args.rasa_config)

        # Listen for messages
        client = mqtt.Client()
        hermes = NluHermesMqtt(
            client,
            args.rasa_url,
            graph_path=args.intent_graph,
            examples_md_path=args.examples_path,
            config_path=args.rasa_config,
            write_graph=args.write_graph,
            word_transform=get_word_transform(args.casing),
            replace_numbers=args.replace_numbers,
            number_language=args.number_language,
            rasa_language=args.rasa_language,
            rasa_project=args.rasa_project,
            rasa_model_dir=args.rasa_model_dir,
            certfile=args.certfile,
            keyfile=args.keyfile,
            siteIds=args.siteId,
            loop=loop,
        )

        _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
        client.connect(args.host, args.port)

        client.connect(args.host, args.port)
        client.loop_start()

        # Run event loop
        hermes.loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")


# -----------------------------------------------------------------------------


def get_word_transform(name: str) -> typing.Callable[[str], str]:
    """Gets a word transformation function by name."""
    if name == "upper":
        return str.upper

    if name == "lower":
        return str.lower

    return lambda s: s


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
