"""Hermes MQTT server for Rasa NLU"""
import asyncio
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
import attr
import networkx as nx
import rhasspynlu
from rhasspyhermes.base import Message
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
from rhasspynlu.intent import Recognition

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

TopicArgs = typing.Mapping[str, typing.Any]
GeneratorType = typing.AsyncIterable[
    typing.Union[Message, typing.Tuple[Message, TopicArgs]]
]


class NluHermesMqtt:
    """Hermes MQTT server for Rasa NLU."""

    def __init__(
        self,
        client,
        rasa_url: str,
        config_path: typing.Optional[Path] = None,
        intent_graph: typing.Optional[nx.DiGraph] = None,
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
        siteIds: typing.Optional[typing.List[str]] = None,
        loop=None,
    ):
        self.client = client
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
        self.siteIds = siteIds or []

        # SSL
        self.ssl_context = ssl.SSLContext()
        if certfile:
            _LOGGER.debug("Using SSL with certfile=%s, keyfile=%s", certfile, keyfile)
            self.ssl_context.load_cert_chain(certfile, keyfile)

        # Async HTTP
        self.loop = loop or asyncio.get_event_loop()
        self.http_session = aiohttp.ClientSession()

        # Create markdown examples
        self.examples_md_path = Path("intent_examples.md")

    # -------------------------------------------------------------------------

    async def handle_query(self, query: NluQuery):
        """Do intent recognition."""
        # try:
        # if not self.intent_graph and self.graph_path and self.graph_path.is_file():
        #     # Load graph from file
        #     with open(self.graph_path, "r") as graph_file:
        #         self.intent_graph = rhasspynlu.json_to_graph(json.load(graph_file))

        # if self.intent_graph:

        #     def intent_filter(intent_name: str) -> bool:
        #         """Filter out intents."""
        #         if query.intentFilter:
        #             return intent_name in query.intentFilter
        #         return True

        #     original_input = query.input

        #     # Replace digits with words
        #     if self.replace_numbers:
        #         # Have to assume whitespace tokenization
        #         words = rhasspynlu.replace_numbers(query.input.split(), self.number_language)
        #         query.input = " ".join(words)

        #     input_text = query.input

        #     # Fix casing for output event
        #     if self.word_transform:
        #         input_text = self.word_transform(input_text)

        #     # Pass in raw query input so raw values will be correct
        #     recognitions = recognize(
        #         query.input,
        #         self.intent_graph,
        #         intent_filter=intent_filter,
        #         word_transform=self.word_transform,
        #         fuzzy=self.fuzzy,
        #     )
        # else:
        #     _LOGGER.error("No intent graph loaded")
        #     recognitions = []

        # if recognitions:
        #     # Use first recognition only.
        #     recognition = recognitions[0]
        #     assert recognition is not None
        #     assert recognition.intent is not None

        #     # intentParsed
        #     self.publish(
        #         NluIntentParsed(
        #             input=input_text,
        #             id=query.id,
        #             siteId=query.siteId,
        #             sessionId=query.sessionId,
        #             intent=Intent(
        #                 intentName=recognition.intent.name,
        #                 confidenceScore=recognition.intent.confidence,
        #             ),
        #             slots=[
        #                 Slot(
        #                     entity=e.entity,
        #                     slotName=e.entity,
        #                     confidence=1,
        #                     value=e.value,
        #                     raw_value=e.raw_value,
        #                     range=SlotRange(
        #                         start=e.start,
        #                         end=e.end,
        #                         raw_start=e.raw_start,
        #                         raw_end=e.raw_end,
        #                     ),
        #                 )
        #                 for e in recognition.entities
        #             ],
        #         )
        #     )

        #     # intent
        #     self.publish(
        #         NluIntent(
        #             input=input_text,
        #             id=query.id,
        #             siteId=query.siteId,
        #             sessionId=query.sessionId,
        #             intent=Intent(
        #                 intentName=recognition.intent.name,
        #                 confidenceScore=recognition.intent.confidence,
        #             ),
        #             slots=[
        #                 Slot(
        #                     entity=e.entity,
        #                     slotName=e.entity,
        #                     confidence=1,
        #                     value=e.value,
        #                     raw_value=e.raw_value,
        #                     range=SlotRange(
        #                         start=e.start,
        #                         end=e.end,
        #                         raw_start=e.raw_start,
        #                         raw_end=e.raw_end,
        #                     ),
        #                 )
        #                 for e in recognition.entities
        #             ],
        #             asrTokens=input_text.split(),
        #             rawAsrTokens=original_input.split(),
        #         ),
        #         intentName=recognition.intent.name,
        #     )
        # else:
        #     # Not recognized
        #     self.publish(
        #         NluIntentNotRecognized(
        #             input=query.input,
        #             id=query.id,
        #             siteId=query.siteId,
        #             sessionId=query.sessionId,
        #         )
        #     )
        # except Exception as e:
        #     _LOGGER.exception("nlu query")
        #     self.publish(
        #         NluError(
        #             siteId=query.siteId,
        #             sessionId=query.sessionId,
        #             error=str(e),
        #             context="",
        #         )
        #     )

    # -------------------------------------------------------------------------

    async def handle_train(
        self, train: NluTrain, siteId: str = "default"
    ) -> typing.Iterable[
        typing.Union[typing.Tuple[NluTrainSuccess, TopicArgs], NluError]
    ]:
        """Transform sentences to intent graph"""
        _LOGGER.debug("<- %s(%s)", train.__class__.__name__, train.id)

        try:
            self.intent_graph = rhasspynlu.json_to_graph(train.graph_dict)

            if self.graph_path:
                # Write graph as JSON
                with open(self.graph_path, "w") as graph_file:
                    json.dump(train.graph_dict, graph_file)

                    _LOGGER.debug("Wrote %s", str(self.graph_path))

            # Build Markdown sentences
            sentences_by_intent = NluHermesMqtt.make_sentences_by_intent(
                self.intent_graph
            )

            # Write to YAML/Markdown file
            with open(self.examples_md_path, "w") as examples_md_file:
                for intent_name, intent_sents in sentences_by_intent.items():
                    # Rasa Markdown training format
                    print(f"## intent:{intent_name}", file=examples_md_file)
                    for intent_sent in intent_sents:
                        raw_index = 0
                        index_entity = {e.raw_start: e for e in intent_sent.entities}
                        entity = None
                        sentence_tokens = []
                        entity_tokens = []
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
                    training_config.write('pipeline: "pretrained_embeddings_spacy"\n')

                # Write markdown directly into YAML.
                # Because reasons.
                with open(self.examples_md_path, "r") as examples_md_file:
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
                training_url = urljoin(self.rasa_url, "model/train")
                training_file.seek(0)
                with open(training_file.name, "rb") as training_data:

                    training_body = {
                        "config": training_config.getvalue(),
                        "nlu": training_data.read().decode("utf-8"),
                    }
                    training_config.close()

                    try:
                        # POST training data
                        async with self.http_session.post(
                            training_url,
                            json=training_body,
                            params=json.dumps({"project": self.rasa_project}),
                            ssl=self.ssl_context,
                        ) as response:
                            response.raise_for_status()

                            model_file = os.path.join(
                                self.rasa_model_dir, response.headers["filename"]
                            )
                            self._logger.debug("Received model %s", model_file)

                            # Replace model with PUT.
                            # Do we really have to do this?
                            model_url = urljoin(self.rasa_url, "model")
                            async with self.http_session.put(
                                model_url, json={"model_file": model_file}
                            ) as response:
                                response.raise_for_status()
                    except Exception:
                        # Rasa gives quite helpful error messages, so extract them from the response.
                        _LOGGER.exception("handle_train")
                        raise Exception(
                            f'{response.reason}: {json.loads(response.content)["message"]}'
                        )

            yield (NluTrainSuccess(id=train.id), {"siteId": siteId})
        except Exception as e:
            yield NluError(
                siteId=siteId, error=str(e), context=train.id, sessionId=train.id
            )

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            topics = [NluQuery.topic()]

            if self.siteIds:
                # Specific siteIds
                topics.extend(
                    [NluTrain.topic(siteId=siteId) for siteId in self.siteIds]
                )
            else:
                # All siteIds
                topics.append(NluTrain.topic(siteId="+"))

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            _LOGGER.debug("Received %s byte(s) on %s", len(msg.payload), msg.topic)
            if msg.topic == NluQuery.topic():
                json_payload = json.loads(msg.payload)

                # Check siteId
                if not self._check_siteId(json_payload):
                    return

                query = NluQuery.from_dict(json_payload)
                _LOGGER.debug("<- %s", query)
                self.publish_all(self.handle_query(query))
            elif NluTrain.is_topic(msg.topic):
                siteId = NluTrain.get_siteId(msg.topic)
                if self.siteIds and (siteId not in self.siteIds):
                    return

                json_payload = json.loads(msg.payload)
                train = NluTrain.from_dict(json_payload)
                self.publish_all(self.handle_train(train, siteId=siteId))
        except Exception:
            _LOGGER.exception("on_message")

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            _LOGGER.debug("-> %s", message)
            topic = message.topic(**topic_args)
            payload = json.dumps(attr.asdict(message))
            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    def publish_all(self, async_generator: GeneratorType):
        """Publish all messages from an async generator"""
        asyncio.run_coroutine_threadsafe(
            self.async_publish_all(async_generator), self.loop
        )

    async def async_publish_all(self, async_generator: GeneratorType):
        """Enumerate all messages in an async generator publish them"""
        async for maybe_message in async_generator:
            if isinstance(maybe_message, Message):
                self.publish(maybe_message)
            else:
                message, kwargs = maybe_message
                self.publish(message, **kwargs)

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True

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

        # TODO: Add converters
        for path in paths:
            _, recognition = rhasspynlu.fsticuffs.path_to_recognition(
                path, intent_graph, extra_converters=extra_converters
            )
            assert recognition, "Path failed"
            sentences_by_intent[recognition.intent.name].append(recognition)

        return sentences_by_intent
