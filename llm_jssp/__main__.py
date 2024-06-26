from pathlib import Path
import click


@click.group()
def main() -> None:
    pass


@main.command(short_help="solve a JSSP problem instance, using a natural language description of the problem")
@click.option("--problemfile", required=True, type=click.Path(exists=True, path_type=Path), help="File containing the description of the problem in English")
@click.option("--flag", is_flag=True, help="describe flag")
@click.argument("FILE", required=True, type=click.Path(exists=True, path_type=Path))
def command(flag: bool, file: Path) -> None:
    pass


if __name__ == "__main__":
    main(prog_name="llm_jssp")
    pass
