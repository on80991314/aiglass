import logging


def setup(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s  %(levelname)-5s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
