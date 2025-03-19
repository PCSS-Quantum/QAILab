#!/bin/sh

echo "\n\033[1m============Preparing developer setup==============\033[0m\n"
{
    uv venv -p3.12 .venv
} || {
    echo "\n\033[1m================uv is not installed================\033[0m\n" &&
    python3 -m pip install uv &&
    python3 -m uv venv -p3.12 .venv
}
source .venv/bin/activate

echo "\n\033[1m==============Installing requirements==============\033[0m\n"
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
uv pip install -r requirements-ql.txt
{
    echo "\n\033[1m===========Adding pre-commit as git hook===========\033[0m\n" &&
    pre-commit install &&

    echo "\n\033[1m===================Running Tests===================\033[0m\n" &&
    pytest tests &&

    echo "\n\033[1m===================Running Linter===================\033[0m\n" &&
    pylint qlearning
} || {
    echo "\033[1m\n==============\033[0;31m!!Something went wrong!!\033[0m\033[1m==============\033[0m\n"
}