"""
Latino Real Estate Market Intelligence Engine
Entry point — run with: python main.py [--level zip|county|state] [--no-growth]
"""

import argparse
import sys
from loguru import logger

from src.config.settings import settings
from src.services.pipeline import Pipeline


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        "logs/pipeline.log",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        compression="zip",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latino Real Estate Market Intelligence Engine"
    )
    parser.add_argument(
        "--level",
        choices=["zip", "county", "state"],
        default=None,
        help="Geographic level (overrides .env GEOGRAPHIC_LEVEL)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="ACS year (overrides .env ACS_YEAR)",
    )
    parser.add_argument(
        "--no-growth",
        action="store_true",
        help="Skip dual-year growth comparison (faster, single-year only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply CLI overrides
    if args.level:
        settings.__dict__["geographic_level"] = args.level
    if args.year:
        settings.__dict__["acs_year"] = args.year

    configure_logging()

    logger.info("=" * 60)
    logger.info("Latino Real Estate Market Intelligence Engine")
    logger.info(f"  Geographic level : {settings.geographic_level}")
    logger.info(f"  ACS year         : {settings.acs_year}")
    logger.info(f"  Previous year    : {settings.acs_year_prev}")
    logger.info(f"  Growth mode      : {'OFF' if args.no_growth else 'ON'}")
    logger.info(f"  Output directory : {settings.output_dir}")
    logger.info("=" * 60)

    pipeline = Pipeline()
    output_paths = pipeline.run(dual_year=not args.no_growth)

    logger.info("\nFiles generated:")
    for fmt, path in output_paths.items():
        logger.info(f"  [{fmt.upper():7}] {path}")


if __name__ == "__main__":
    main()
