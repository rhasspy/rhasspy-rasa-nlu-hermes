"""Hermes MQTT server for Rasa NLU"""
import gzip
import io
import json
import logging
import os
import random
import ssl
import tempfile
import typing
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
import networkx as nx
import rhasspynlu
from rhasspyhermes.base import Message
from rhasspyhermes.client import GeneratorType, HermesClient, TopicArgs
from rhasspyhermes.intent import Intent, Slot, SlotRange
from rhasspyhermes.nlu import (
    NluError,
    NluIntent,
    NluIntentNotRecognized,
    NluIntentParsed,
    NluQuery,
    NluTrain,
    NluTrainSuccess,
)
from rhasspynlu import Sentence
from rhasspynlu.intent import Entity, Recognition

_LOGGER = logging.getLogger("rhasspyrasa_nlu_hermes")

# -----------------------------------------------------------------------------


class NluHermesMqtt(HermesClient):
    """Hermes MQTT server for Rasa NLU."""

    def __init__(
        self,
        client,
        rasa_url: str,
        config_path: typing.Optional[Path] = None,
        intent_graph: typing.Optional[nx.DiGraph] = None,
        examples_md_path: typing.Optional[Path] = None,
        graph_path: typing.Optional[Path] = None,
        write_graph: bool = False,
        default_entities: typing.Dict[str, typing.Iterable[Sentence]] = None,
        word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        replace_numbers: bool = False,
        number_language: typing.Optional[str] = None,
        rasa_language: str = "en",
        rasa_project: str = "rhasspy",
        rasa_model_dir: str = "models",
        certfile: typing.Optional[str] = None,
        keyfile: typing.Optional[str] = None,
        site_ids: typing.Optional[typing.List[str]] = None,
    ):
        super().__init__("rhasspyrasa_nlu_hermes", client, site_ids=site_ids)

        self.subscribe(NluQuery, NluTrain)

        self.rasa_url = rasa_url
        self.graph_path = graph_path
        self.config_path = config_path
        self.intent_graph = intent_graph
        self.write_graph = write_graph
        self.default_entities = default_entities or {}
        self.word_transform = word_transform
        self.replace_numbers = replace_numbers
        self.number_language = number_language
        self.rasa_language = rasa_language
        self.rasa_project = rasa_project
        self.rasa_model_dir = rasa_model_dir

        # SSL
        self.ssl_context = ssl.SSLContext()
        if certfile:
            _LOGGER.debug("Using SSL with certfile=%s, keyfile=%s", certfile, keyfile)
            self.ssl_context.load_cert_chain(certfile, keyfile)

        # Async HTTP
        self._http_session: typing.Optional[aiohttp.ClientSession] = None

        # Create markdown examples
        self.examples_md_path = examples_md_path

    @property
    def http_session(self):
        """Get or create async HTTP session"""
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()

        return self._http_session

    # -------------------------------------------------------------------------

    async def handle_query(
        self, query: NluQuery
    ) -> typing.AsyncIterable[
        typing.Union[
            NluIntentParsed,
            typing.Tuple[NluIntent, TopicArgs],
            NluIntentNotRecognized,
            NluError,
        ]
    ]:
        """Do intent recognition."""
        try:
            original_input = query.input

            # Replace digits with words
            if self.replace_numbers:
                # Have to assume whitespace tokenization
                words = rhasspynlu.replace_numbers(
                    query.input.split(), self.number_language
                )
                query.input = " ".join(words)

            input_text = query.input

            # Fix casing for output event
            if self.word_transform:
                input_text = self.word_transform(input_text)

            parse_url = urljoin(self.rasa_url, "model/parse")
            _LOGGER.debug(parse_url)

            async with self.http_session.post(
                parse_url,
                json={"text": input_text, "project": self.rasa_project},
                ssl=self.ssl_context,
            ) as response:
                response.raise_for_status()
                intent_json = await response.json()
                intent = intent_json.get("intent", {})
                intent_name = intent.get("name", "")

                if intent_name and (
                    query.intent_filter is None or intent_name in query.intent_filter
                ):
                    confidence_score = float(intent.get("confidence", 0.0))
                    slots = [
                        Slot(
                            entity=e.get("entity", ""),
                            slot_name=e.get("entity", ""),
                            confidence=float(e.get("confidence", 0.0)),
                            value={"kind": "Unknown", "value": e.get("value", "")},
                            raw_value=e.get("value", ""),
                            range=SlotRange(
                                start=int(e.get("start", 0)),
                                end=int(e.get("end", 1)),
                                raw_start=int(e.get("start", 0)),
                                raw_end=int(e.get("end", 1)),
                            ),
                        )
                        for e in intent_json.get("entities", [])
                    ]

                    # intentParsed
                    yield NluIntentParsed(
                        input=input_text,
                        id=query.id,
                        site_id=query.site_id,
                        session_id=query.session_id,
                        intent=Intent(
                            intent_name=intent_name, confidence_score=confidence_score
                        ),
                        slots=slots,
                    )

                    # intent
                    yield (
                        NluIntent(
                            input=input_text,
                            id=query.id,
                            site_id=query.site_id,
                            session_id=query.session_id,
                            intent=Intent(
                                intent_name=intent_name,
                                confidence_score=confidence_score,
                            ),
                            slots=slots,
                            asr_tokens=[NluIntent.make_asr_tokens(input_text.split())],
                            raw_input=original_input,
                            lang=query.lang,
                        ),
                        {"intent_name": intent_name},
                    )
                else:
                    # Not recognized
                    yield NluIntentNotRecognized(
                        input=query.input,
                        id=query.id,
                        site_id=query.site_id,
                        session_id=query.session_id,
                    )
        except Exception as e:
            _LOGGER.exception("nlu query")
            yield NluError(
                site_id=query.site_id,
                session_id=query.session_id,
                error=str(e),
                context=query.input,
            )

    # -------------------------------------------------------------------------

    async def handle_train(
        self, train: NluTrain, site_id: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[NluTrainSuccess, TopicArgs], NluError]
    ]:
        """Transform sentences to intent graph"""
        try:
            _LOGGER.debug("Loading %s", train.graph_path)
            with gzip.GzipFile(train.graph_path, mode="rb") as graph_gzip:
                self.intent_graph = nx.readwrite.gpickle.read_gpickle(graph_gzip)

            # Build Markdown sentences
            sentences_by_intent = NluHermesMqtt.make_sentences_by_intent(
                self.intent_graph
            )

            if self.examples_md_path is not None:
                # Use user-specified file
                examples_md_file = open(self.examples_md_path, "w+")
            else:
                # Use temporary file
                examples_md_file = typing.cast(
                    typing.TextIO, tempfile.TemporaryFile(mode="w+")
                )

            with examples_md_file:
                # Write to YAML/Markdown file
                for intent_name, intent_sents in sentences_by_intent.items():
                    # Rasa Markdown training format
                    print(f"## intent:{intent_name}", file=examples_md_file)
                    for intent_sent in intent_sents:
                        raw_index = 0
                        index_entity = {e.raw_start: e for e in intent_sent.entities}
                        entity: typing.Optional[Entity] = None
                        sentence_tokens: typing.List[str] = []
                        entity_tokens: typing.List[str] = []
                        for raw_token in intent_sent.raw_tokens:
                            token = raw_token
                            if entity and (raw_index >= entity.raw_end):
                                # Finish current entity
                                last_token = entity_tokens[-1]
                                entity_tokens[-1] = f"{last_token}]({entity.entity})"
                                sentence_tokens.extend(entity_tokens)
                                entity = None
                                entity_tokens = []

                            new_entity = index_entity.get(raw_index)
                            if new_entity:
                                # Begin new entity
                                assert entity is None, "Unclosed entity"
                                entity = new_entity
                                entity_tokens = []
                                token = f"[{token}"

                            if entity:
                                # Add to current entity
                                entity_tokens.append(token)
                            else:
                                # Add directly to sentence
                                sentence_tokens.append(token)

                            raw_index += len(raw_token) + 1

                        if entity:
                            # Finish final entity
                            last_token = entity_tokens[-1]
                            entity_tokens[-1] = f"{last_token}]({entity.entity})"
                            sentence_tokens.extend(entity_tokens)

                        # Print single example
                        print("-", " ".join(sentence_tokens), file=examples_md_file)

                    # Newline between intents
                    print("", file=examples_md_file)

                # Create training YAML file
                with tempfile.NamedTemporaryFile(
                    suffix=".json", mode="w+", delete=False
                ) as training_file:

                    training_config = io.StringIO()

                    if self.config_path:
                        # Use provided config
                        with open(self.config_path, "r") as config_file:
                            # Copy verbatim
                            for line in config_file:
                                training_config.write(line)
                    else:
                        # Use default config
                        training_config.write(f'language: "{self.rasa_language}"\n')
                        training_config.write(
                            'pipeline: "pretrained_embeddings_spacy"\n'
                        )

                    # Write markdown directly into YAML.
                    # Because reasons.
                    examples_md_file.seek(0)
                    blank_line = False
                    for line in examples_md_file:
                        line = line.strip()
                        if line:
                            if blank_line:
                                print("", file=training_file)
                                blank_line = False

                            print(f"  {line}", file=training_file)
                        else:
                            blank_line = True

                    # Do training via HTTP API
                    training_file.seek(0)
                    with open(training_file.name, "rb") as training_data:

                        training_body = {
                            "config": training_config.getvalue(),
                            "nlu": training_data.read().decode("utf-8"),
                        }
                        training_config.close()

                        # POST training data
                        training_response: typing.Optional[bytes] = None

                        try:
                            training_url = urljoin(self.rasa_url, "model/train")
                            _LOGGER.debug(training_url)
                            async with self.http_session.post(
                                training_url,
                                json=training_body,
                                params=json.dumps(
                                    {"project": self.rasa_project}, ensure_ascii=False
                                ),
                                ssl=self.ssl_context,
                            ) as response:
                                training_response = await response.read()
                                response.raise_for_status()

                                model_file = os.path.join(
                                    self.rasa_model_dir, response.headers["filename"]
                                )
                                _LOGGER.debug("Received model %s", model_file)

                                # Replace model with PUT.
                                # Do we really have to do this?
                                model_url = urljoin(self.rasa_url, "model")
                                _LOGGER.debug(model_url)
                                async with self.http_session.put(
                                    model_url, json={"model_file": model_file}
                                ) as response:
                                    response.raise_for_status()
                        except Exception as e:
                            if training_response:
                                _LOGGER.exception("rasa train")

                                # Rasa gives quite helpful error messages, so extract them from the response.
                                error_message = json.loads(training_response)["message"]
                                raise Exception(f"{response.reason}: {error_message}")

                            # Empty response; re-raise exception
                            raise e

            yield (NluTrainSuccess(id=train.id), {"site_id": site_id})
        except Exception as e:
            _LOGGER.exception("handle_train")
            yield NluError(
                site_id=site_id, error=str(e), context=train.id, session_id=train.id
            )

    # -------------------------------------------------------------------------

    async def on_message(
        self,
        message: Message,
        site_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker."""
        if isinstance(message, NluQuery):
            async for query_result in self.handle_query(message):
                yield query_result
        elif isinstance(message, NluTrain):
            assert site_id, "Missing site_id"
            async for train_result in self.handle_train(message, site_id=site_id):
                yield train_result
        else:
            _LOGGER.warning("Unexpected message: %s", message)

    # -------------------------------------------------------------------------

    @classmethod
    def make_sentences_by_intent(
        cls,
        intent_graph: nx.DiGraph,
        num_samples: typing.Optional[int] = None,
        extra_converters=None,
    ) -> typing.Dict[str, typing.List[Recognition]]:
        """Get all sentences from a graph."""

        sentences_by_intent: typing.Dict[str, typing.List[Recognition]] = defaultdict(
            list
        )

        start_node = None
        end_node = None
        for node, node_data in intent_graph.nodes(data=True):
            if node_data.get("start", False):
                start_node = node
            elif node_data.get("final", False):
                end_node = node

            if start_node and end_node:
                break

        assert (start_node is not None) and (
            end_node is not None
        ), "Missing start/end node(s)"

        if num_samples is not None:
            # Randomly sample
            paths = random.sample(
                list(nx.all_simple_paths(intent_graph, start_node, end_node)),
                num_samples,
            )
        else:
            # Use generator
            paths = nx.all_simple_paths(intent_graph, start_node, end_node)

        for path in paths:
            _, recognition = rhasspynlu.fsticuffs.path_to_recognition(
                path, intent_graph, extra_converters=extra_converters
            )
            assert recognition, "Path failed"
            if recognition.intent:
                sentences_by_intent[recognition.intent.name].append(recognition)

        return sentences_by_intent
