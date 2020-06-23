# Rhasspy Rasa NLU Hermes Service

MQTT service for Rhasspy that uses [Rasa NLU](https://rasa.com/) to do intent recognition.

## Requirements

* Python 3.7
* A [Rasa NLU](https://rasa.com/) server

## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-rasa-nlu-hermes
$ cd rhasspy-rasa-nlu-hermes
$ ./configure
$ make
$ make install
```

## Running

```bash
$ bin/rhasspy-rasa-nlu-hermes <ARGS>
```

## Command-Line Options

```
usage: rhasspy-rasa-nlu-hermes [-h] --rasa-url RASA_URL
                               [--intent-graph INTENT_GRAPH]
                               [--examples-path EXAMPLES_PATH]
                               [--rasa-config RASA_CONFIG]
                               [--rasa-project RASA_PROJECT]
                               [--rasa-model-dir RASA_MODEL_DIR]
                               [--rasa-language RASA_LANGUAGE] [--write-graph]
                               [--casing {upper,lower,ignore}]
                               [--replace-numbers]
                               [--number-language NUMBER_LANGUAGE]
                               [--certfile CERTFILE] [--keyfile KEYFILE]
                               [--host HOST] [--port PORT]
                               [--username USERNAME] [--password PASSWORD]
                               [--tls] [--tls-ca-certs TLS_CA_CERTS]
                               [--tls-certfile TLS_CERTFILE]
                               [--tls-keyfile TLS_KEYFILE]
                               [--tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}]
                               [--tls-version TLS_VERSION]
                               [--tls-ciphers TLS_CIPHERS] [--site-id SITE_ID]
                               [--debug] [--log-format LOG_FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  --rasa-url RASA_URL   URL of Rasa NLU server
  --intent-graph INTENT_GRAPH
                        Path to rhasspy intent graph JSON file
  --examples-path EXAMPLES_PATH
                        Path to write examples markdown file
  --rasa-config RASA_CONFIG
                        Path to Rasa NLU's config.yml file
  --rasa-project RASA_PROJECT
                        Project name used when training Rasa NLU (default:
                        rhasspy)
  --rasa-model-dir RASA_MODEL_DIR
                        Directory name where Rasa NLU stores its model files
                        (default: models)
  --rasa-language RASA_LANGUAGE
                        Language used for Rasa NLU training (default: en)
  --write-graph         Write training graph to intent-graph path
  --casing {upper,lower,ignore}
                        Case transformation for input text (default: ignore)
  --replace-numbers     Replace digits with words in queries (75 -> seventy
                        five)
  --number-language NUMBER_LANGUAGE
                        Language/locale used for number replacement (default:
                        en)
  --certfile CERTFILE   SSL certificate file
  --keyfile KEYFILE     SSL private key file (optional)
  --host HOST           MQTT host (default: localhost)
  --port PORT           MQTT port (default: 1883)
  --username USERNAME   MQTT username
  --password PASSWORD   MQTT password
  --tls                 Enable MQTT TLS
  --tls-ca-certs TLS_CA_CERTS
                        MQTT TLS Certificate Authority certificate files
  --tls-certfile TLS_CERTFILE
                        MQTT TLS certificate file (PEM)
  --tls-keyfile TLS_KEYFILE
                        MQTT TLS key file (PEM)
  --tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}
                        MQTT TLS certificate requirements (default:
                        CERT_REQUIRED)
  --tls-version TLS_VERSION
                        MQTT TLS version (default: highest)
  --tls-ciphers TLS_CIPHERS
                        MQTT TLS ciphers to use
  --site-id SITE_ID     Hermes site id(s) to listen for (default: all)
  --debug               Print DEBUG messages to the console
  --log-format LOG_FORMAT
                        Python logger format
```
