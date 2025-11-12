import logging
import time
from collections.abc import MutableMapping
from concurrent.futures import wait
from pathlib import Path
from typing import Annotated, Any, TypedDict

import click
from click.testing import CliRunner
from yaml import Loader, load

from .. import config
from ..cli.cli import (
    cli,
    click_parameter_decorators_from_typed_dict,
)

log = logging.getLogger(__name__)


def click_get_defaults_from_file(ctx, param, value):  # noqa: ANN001, ARG001
    if not value:
        raise click.MissingParameter
    path = Path(value)
    input_file = path if path.exists() else Path(config.CONFIG_LOCAL_DIR, path)
    try:
        with input_file.open() as f:
            _config: dict[str, list[dict[str, Any]]] = load(f.read(), Loader=Loader)  # noqa: S506
            ctx.default_map = _config
    except Exception as e:
        msg = f"Failed to load batch config file: {e}"
        raise click.BadParameter(msg) from e
    return value


class BatchCliTypedDict(TypedDict):
    batch_config_file: Annotated[
        bool,
        click.option(
            "--batch-config-file",
            type=click.Path(),
            callback=click_get_defaults_from_file,
            is_eager=True,
            expose_value=False,
            help="Read batch configuration from yaml file",
        ),
    ]


def build_sub_cmd_args(batch_config: MutableMapping[str, Any] | None):
    bool_options = {
        "drop_old": True,
        "load": True,
        "rebuild_index": False,
        "search_serial": True,
        "search_concurrent": True,
        "dry_run": False,
        "custom_dataset_use_shuffled": True,
        "custom_dataset_with_gt": True,
    }

    def format_option(key: str, value: Any):
        opt_name = key.replace("_", "-")

        if key in bool_options:
            return format_bool_option(opt_name, value, skip=False)

        if key.startswith("skip_"):
            raw_key = key[5:]
            raw_opt = raw_key.replace("_", "-")
            return format_bool_option(raw_opt, value, skip=True, raw_key=raw_key)

        return [f"--{opt_name}", str(value)]

    def format_bool_option(opt_name: str, value: Any, skip: bool = False, raw_key: str | None = None):
        if isinstance(value, bool):
            if skip:
                if bool_options.get(raw_key, False):
                    return [f"--skip-{opt_name}"] if value else [f"--{opt_name}"]
                return [f"--{opt_name}", str(value)]
            if value:
                return [f"--{opt_name}"]
            if bool_options.get(opt_name.replace("-", "_"), False):
                return [f"--skip-{opt_name}"]
            return []
        return [f"--{opt_name}", str(value)]

    args_arr = []
    for sub_cmd_key, sub_cmd_config_list in batch_config.items():
        for sub_cmd_args in sub_cmd_config_list:
            args = [sub_cmd_key]
            for k, v in sub_cmd_args.items():
                args.extend(format_option(k, v))
            args_arr.append(args)

    return args_arr


@cli.command()
@click_parameter_decorators_from_typed_dict(BatchCliTypedDict)
def BatchCli():
    ctx = click.get_current_context()
    batch_config = ctx.default_map

    runner = CliRunner()

    args_arr = build_sub_cmd_args(batch_config)

    for args in args_arr:
        log.info(f"got batch config: {' '.join(args)}")

    for args in args_arr:
        result = runner.invoke(cli, args)

        from ..interface import benchmark_runner, global_result_future

        # Wait for the background task to complete
        if global_result_future:
            wait([global_result_future])

        # Poll until the running task is cleared, processing signals
        max_wait = 30  # Maximum 30 seconds
        waited = 0
        while benchmark_runner.running_task is not None and waited < max_wait:
            # Call _try_get_signal to process completion signals from the pipe
            benchmark_runner._try_get_signal()
            if benchmark_runner.running_task is None:
                break
            time.sleep(0.5)
            waited += 0.5

        if benchmark_runner.running_task is not None:
            log.warning(f"Task did not clear after {max_wait} seconds, proceeding anyway")

        # Additional small delay to ensure clean state
        time.sleep(2)

        if result.exception:
            log.exception(f"failed to run sub command: {args[0]}", exc_info=result.exception)
