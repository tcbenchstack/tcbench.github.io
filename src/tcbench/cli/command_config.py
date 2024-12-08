from __future__ import annotations
import rich_click as click

import tcbench
from tcbench.cli import console

@click.group()
@click.pass_context
def config(ctx):
    """Show/Change TBBench global configurations."""
    pass

def _parse_set_option(option_values):
    if len(option_values) != 1 or "=" not in option_values[0]:
        raise click.BadParameter(
            "Invalid syntax. Provide a single value using the syntax: "
            "config.param.name=value"
        )
    param_name, param_value = option_values[0].split("=")
    if not tcbench.is_valid_config(param_name, param_value):
        raise click.BadParameter(
            "Unrecognized option or wrong value. Provide a single value "
            "using the syntax: config.param.name=value"
        )
    return param_name, param_value


@config.command(name="set")
@click.pass_context
@click.argument('option', nargs=-1, type=click.UNPROCESSED)
def _set(ctx, option):
    """Set a TCBench config to a specific value."""
    param_name, param_value = _parse_set_option(option)
    tcbenchrc = tcbench.get_config()
    tcbenchrc[param_name] = param_value
    tcbenchrc.save()
    console.print("Configuration updated!")

@config.command(name="init")
@click.pass_context
def _reset(ctx):
    """Reset all TCBench configurations to default values."""
    tcbench.init_tcbenchrc()
    console.print("Configuration reset to default!")

@config.command(name="show")
@click.pass_context
def show(ctx):
    """Show current configrations."""
    with open(tcbench.TCBENCHRC_PATH) as fin:
        text = fin.read()
    console.print(text)
