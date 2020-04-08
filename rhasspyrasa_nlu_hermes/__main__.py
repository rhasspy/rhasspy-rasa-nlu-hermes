"""Hermes MQTT service for Rasa NLU"""
import argparse
import asyncio
import logging
import typing
from pathlib import Path

import paho.mqtt.client as mqtt
import rhasspyhermes.cli as hermes_cli

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

    hermes_cli.add_hermes_args(parser)
    args = parser.parse_args()

    hermes_cli.setup_logging(args)
    _LOGGER.debug(args)

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
        site_ids=args.site_id,
    )

    _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
    hermes_cli.connect(client, args)
    client.loop_start()

    try:
        # Run event loop
        asyncio.run(hermes.handle_messages_async())
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")
        client.loop_stop()


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
