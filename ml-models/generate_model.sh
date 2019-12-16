#! /bin/bash

docker-compose run --rm flask /bin/bash -c "cd ml-models/ && python ./generate_model.py"
